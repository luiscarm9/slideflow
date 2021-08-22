import os
import types
import shutil
import logging
import itertools
import csv
import queue, threading
import time
import pickle
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.colors as mcol

from os.path import join, exists, isdir, dirname
from glob import glob
from random import shuffle
from multiprocessing.dummy import Pool as DPool
from functools import partial
from datetime import datetime
from PIL import Image
from io import BytesIO
from statistics import mean, median

import slideflow as sf
import slideflow.util as sfutil
import slideflow.io as sfio

from slideflow import project_utils
from slideflow.io import Dataset
from slideflow.statistics import TFRecordMap, calculate_centroid
from slideflow.util import TCGA, ProgressBar, log, StainNormalizer

NO_LABEL = 'no_label'
SILENT = 'SILENT'
SOURCE_DIR = os.path.dirname(os.path.realpath(__file__))
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DEFAULT_FLAGS = {
	'skip_verification': False,
	'eval_batch_size': 64,
	'num_threads': 4,
	'logging_levels': {
		'info': 3,
		'warn': 3,
		'error': 3,
		'complete': 3,
		'silent': False
	}
}

class SlideflowProject:

	def __init__(self, project_folder, gpu=None, gpu_pool=None, reverse_select_gpu=True, 
					interactive=True, flags=None):
		'''Initializes project by creating project folder, prompting user for project settings, and project
		settings to "settings.json" within the project directory.
		
		Args:
			project_folder:		Project folder
			gpu_pool:			Number of available GPUs. Will try to autoselect GPU if provided
			reverse_select_gpu:	Will try to select GPU from available pool in reverse
			gpu:				Int, forces GPU selection to the indicated GPU number.
			interactive:		Bool, if true, will solicit project information from the user
									via text prompts if if the project has not yet been initialized
		'''
		
		# Configure flags and logging
		self.FLAGS = flags if flags else DEFAULT_FLAGS
		log.configure(levels=self.FLAGS['logging_levels'])

		log.header(f"Slideflow v{sf.__version__}\n================")
		log.header("Loading project...")

		if project_folder and not os.path.exists(project_folder):
			if interactive:
				if sfutil.yes_no_input(f'Directory "{project_folder}" does not exist. Create directory and set as project root? [Y/n] ', default='yes'):
					os.makedirs(project_folder)
				else:
					project_folder = sfutil.dir_input("Where is the project root directory? ", 
													  None, 
													  create_on_invalid=True, 
													  absolute=True)
			else:
				log.info(f"Project directory {project_folder} not found; will create.")
				os.makedirs(project_folder)
		if not project_folder:
			project_folder = sfutil.dir_input("Where is the project root directory? ", 
											  None, 
											  create_on_invalid=True, 
											  absolute=True)

		log.configure(filename=join(project_folder, "log.log"))

		if exists(join(project_folder, "settings.json")):
			self.load_project(project_folder)
		elif interactive:
			self.create_project(project_folder)

		# Set up GPU
		if gpu is not None:
			self.select_gpu(gpu)
		elif gpu_pool:
			self.autoselect_gpu(gpu_pool, reverse=reverse_select_gpu)

	def autoselect_gpu(self, number_available, reverse=True):
		'''Automatically claims a free GPU.
		
		Args:
			number_available:	Total number of GPUs available to select from
			reverse:			Bool, if True, will select GPU from pool in reverse
		'''
		log.header("Selecting GPU...")

		if not number_available:
			os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
			log.warn(f"Disabling GPU access.")
		else:
			gpus = range(number_available) if not reverse else reversed(range(number_available))
			gpu_selected = -1
			if len(gpus):
				gpu_selected = gpus[0]
				os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_selected)
				log.info(f'Using GPU {gpu_selected}')

	def select_gpu(self, gpu):
		'''Sets environmental variables such that the indicated GPU is used by CUDA/Tensorflow.'''
		if gpu == -1:
			os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
			log.warn(f"Disabling GPU access.")
		else:
			log.empty(f"Using GPU #{gpu}", 1)
			os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

		import tensorflow as tf
		
	def _get_hp(self, row, header):
		'''Internal function used to convert a row in the batch_train CSV file into a HyperParameters object.'''
		from slideflow.model import HyperParameters

		model_name_i = header.index('model_name')
		args = header[0:model_name_i] + header[model_name_i+1:]
		model_name = row[model_name_i]
		hp = HyperParameters()
		for arg in args:
			value = row[header.index(arg)]
			if arg in hp._get_args():
				if arg != 'finetune_epochs':
					arg_type = type(getattr(hp, arg))
					if arg_type == bool:
						if value.lower() in ['true', 'yes', 'y', 't']:
							bool_val = True
						elif value.lower() in ['false', 'no', 'n', 'f']:
							bool_val = False
						else:
							log.warn(f'Unable to parse arg "{arg}" with value "{value}" in batch training file into true/false; will default to True', 1)
							bool_val = True
						setattr(hp, arg, bool_val)
					else:
						setattr(hp, arg, arg_type(value))
				else:
					epochs = [int(i) for i in value.translate(str.maketrans({'[':'', ']':''})).split(',')]
					setattr(hp, arg, epochs)
			else:
				log.error(f"Unknown argument '{arg}' found in training config file.", 0)
		return hp, model_name

	def _get_hyperparameter_combinations(self, hyperparameters, models, batch_train_file):
		'''Internal function to organize list of hyperparameters ojects and associated models names.
		
		Args:
			hyperparameters:		List of Hyperparameters objects
			models:					List of model names
			batch_train_file:		Path to train train TSV file

		Returns:
			List of (Hyperparameter, model_name) for each HP combination
		'''
		from slideflow.model import HyperParameterError

		if not hyperparameters:
			hp_models_to_train = self._get_valid_models(batch_train_file, models)
		else:
			hp_models_to_train = [models]

		hyperparameter_list = []	
		if not hyperparameters:
			# Assembling list of models and hyperparameters from batch_train.tsv file
			batch_train_rows = []
			with open(batch_train_file) as csv_file:
				reader = csv.reader(csv_file, delimiter='\t')
				header = next(reader)
				for row in reader:
					batch_train_rows += [row]
				
			for row in batch_train_rows:
				# Read hyperparameters
				try:
					hp, hp_model_name = self._get_hp(row, header)
				except HyperParameterError as e:
					log.error("Invalid Hyperparameter combination: " + str(e))
					return

				if hp_model_name not in hp_models_to_train: continue

				hyperparameter_list += [[hp, hp_model_name]]
		elif isinstance(hyperparameters, list) and isinstance(models, list):
			if len(models) != len(hyperparameters):
				log.error(f"Unable to iterate through hyperparameters provided; length of hyperparameters ({len(hyperparameters)}) much match length of models ({len(models)})", 1)
				return
			for i in range(len(models)):
				if not hyperparameters[i].validate():
					return
				hyperparameter_list += [[hyperparameters[i], models[i]]]
		else:
			if not hyperparameters.validate():
				return
			hyperparameter_list = [[hyperparameters, models]]
		return hyperparameter_list

	def _get_valid_models(self, batch_train_file, models):
		'''Internal function used to scan a batch_train file for valid, trainable models.'''
		models_to_train = []
		with open(batch_train_file) as csv_file:
			reader = csv.reader(csv_file, delimiter="\t")
			header = next(reader)
			try:
				model_name_i = header.index('model_name')
			except:
				err_msg = "Unable to find column 'model_name' in the batch training config file."
				log.error(err_msg)
				raise ValueError(err_msg)
			for row in reader:
				model_name = row[model_name_i]
				# First check if this row is a valid model
				if (not models) or (isinstance(models, str) and model_name==models) or model_name in models:
					# Now verify there are no duplicate model names
					if model_name in models_to_train:
						err_msg = f"Duplicate model names found in {sfutil.green(batch_train_file)}."
						log.error(err_msg)
						raise ValueError(err_msg)
					models_to_train += [model_name]
		return models_to_train

	def add_dataset(self, name, slides, roi, tiles, tfrecords, path=None):
		'''Adds a dataset to the dataset configuration file.

		Args:
			name:		Dataset name.
			slides:		Path to directory containing slides.
			roi:		Path to directory containing CSV ROIs.
			tiles:		Path to directory in which to store extracted tiles.
			tfrecords:	Path to directory in which to store TFRecords of extracted tiles.
			path:		(optional) Path to dataset configuration file. If not provided, uses project default.
		'''

		if not path:
			path = self.PROJECT['dataset_config']
		try:
			datasets_data = sfutil.load_json(path)
		except FileNotFoundError:
			datasets_data = {}
		datasets_data.update({name: {
			'slides': slides,
			'roi': roi,
			'tiles': tiles,
			'tfrecords': tfrecords,
		}})
		sfutil.write_json(datasets_data, path)
		log.info(f"Saved dataset {name} to {path}")

	def associate_slide_names(self):
		'''Funtion to automatically associate patient names with slide filenames in the annotations file.'''
		log.header("Associating slide names...")
		dataset = self.get_dataset(tile_px=0, tile_um=0, verification=None)
		dataset.update_annotations_with_slidenames(self.PROJECT['annotations'])

	def create_blank_annotations_file(self, filename=None):
		'''Creates an example blank annotations file.'''
		if not filename: 
			filename = self.PROJECT['annotations']
		with open(filename, 'w') as csv_outfile:
			csv_writer = csv.writer(csv_outfile, delimiter=',')
			header = [TCGA.patient, 'dataset', 'category']
			csv_writer.writerow(header)

	def create_blank_train_config(self, filename=None):
		'''Creates a CSV file with the batch training hyperparameter structure.'''
		from slideflow.model import HyperParameters

		if not filename:
			filename = self.PROJECT['batch_train_config']
		with open(filename, 'w') as csv_outfile:
			writer = csv.writer(csv_outfile, delimiter='\t')
			# Create headers and first row
			header = ['model_name']
			firstrow = ['model1']
			default_hp = HyperParameters()
			for arg in default_hp._get_args():
				header += [arg]
				firstrow += [getattr(default_hp, arg)]
			writer.writerow(header)
			writer.writerow(firstrow)

	def create_hyperparameter_sweep(self, tile_px, tile_um, finetune_epochs, label=None, filename=None, **kwargs):
		'''Prepares a hyperparameter sweep, saving to a batch train TSV file.'''
		log.header("Preparing hyperparameter sweep...")
		pdict = kwargs
		pdict.update({'tile_px': tile_px, 'tile_um': tile_um})

		args = list(pdict.keys())
		for arg in args:
			if not isinstance(pdict[arg], list):
				pdict[arg] = [pdict[arg]]
		argsv = list(pdict.values())
		sweep = list(itertools.product(*argsv))

		from slideflow.model import HyperParameters

		if not filename:
			filename = self.PROJECT['batch_train_config']
		label = '' if not label else f'{label}-'
		with open(filename, 'w') as csv_outfile:
			writer = csv.writer(csv_outfile, delimiter='\t')
			# Create headers
			header = ['model_name', 'finetune_epochs']
			for arg in args:
				header += [arg]
			writer.writerow(header)
			# Iterate through sweep
			for i, params in enumerate(sweep):
				row = [f'{label}HPSweep{i}', ','.join([str(f) for f in finetune_epochs])]
				full_params = dict(zip(['finetune_epochs'] + args, [finetune_epochs] + list(params)))
				hp = HyperParameters(**full_params)
				for arg in args:
					row += [getattr(hp, arg)]
				writer.writerow(row)
		log.complete(f"Wrote {len(sweep)} combinations for sweep to {sfutil.green(filename)}")

	def create_project(self, project_folder):
		'''Prompts user to provide all relevant project configuration and saves configuration to "settings.json".'''
		# General setup and slide configuration
		project = {
			'root': project_folder,
			'slideflow_version': sf.__version__
		}
		project['name'] = input("What is the project name? ")
		
		# Ask for annotations file location; if one has not been made, offer to create a blank template and then exit
		if not sfutil.yes_no_input("Has an annotations (CSV) file already been created? [y/N] ", default='no'):
			if sfutil.yes_no_input("Create a blank annotations file? [Y/n] ", default='yes'):
				project['annotations'] = sfutil.file_input("Where will the annotation file be located? [./annotations.csv] ", 
									root=project['root'], default='./annotations.csv', filetype="csv", verify=False)
				self.create_blank_annotations_file(project['annotations'])
		else:
			project['annotations'] = sfutil.file_input("Where is the project annotations (CSV) file located? [./annotations.csv] ", 
									root=project['root'], default='./annotations.csv', filetype="csv")

		# Dataset configuration
		project['dataset_config'] = sfutil.file_input("Where is the dataset configuration file located? [./datasets.json] ",
													root=project['root'], default='./datasets.json', filetype='json', verify=False)

		project['datasets'] = []
		while not project['datasets']:
			datasets_data, datasets_names = self.load_datasets(project['dataset_config'])

			print(sfutil.bold("Detected datasets:"))
			if not len(datasets_names):
				print(" [None]")
			else:
				for i, name in enumerate(datasets_names):
					print(f" {i+1}. {name}")
				print(f" {len(datasets_names)+1}. ADD NEW")
				dataset_selection = sfutil.choice_input(f"Which datasets should be used? (choose {len(datasets_names)+1} to add a new dataset) ", valid_choices=[str(l) for l in list(range(1, len(datasets_names)+2))], multi_choice=True)

			if not len(datasets_names) or str(len(datasets_names)+1) in dataset_selection:
				# Create new dataset
				print(f"{sfutil.bold('Creating new dataset')}")
				dataset_name = input("What is the dataset name? ")
				dataset_slides = sfutil.dir_input("Where are the slides stored? [./slides] ",
										root=project['root'], default='./slides', create_on_invalid=True)
				dataset_roi = sfutil.dir_input("Where are the ROI files (CSV) stored? [./slides] ",
										root=project['root'], default='./slides', create_on_invalid=True)
				dataset_tiles = sfutil.dir_input("Where will the tessellated image tiles be stored? (recommend SSD) [./tiles] ",
										root=project['root'], default='./tiles', create_on_invalid=True)
				dataset_tfrecords = sfutil.dir_input("Where should the TFRecord files be stored? (recommend HDD) [./tfrecord] ",
										root=project['root'], default='./tfrecord', create_on_invalid=True)

				self.add_dataset(name=dataset_name,
								 slides=dataset_slides,
								 roi=dataset_roi,
								 tiles=dataset_tiles,
								 tfrecords=dataset_tfrecords,
								 path=project['dataset_config'])

				print("Updated dataset configuration file.")
			else:
				try:
					project['datasets'] = [datasets_names[int(j)-1] for j in dataset_selection]
				except TypeError:
					print(f'Invalid selection: {dataset_selection}')
					continue

		# Training
		project['models_dir'] = sfutil.dir_input("Where should the saved models be stored? [./models] ",
									root=project['root'], default='./models', create_on_invalid=True)
		project['use_fp16'] = sfutil.yes_no_input("Should FP16 be used instead of FP32? (recommended) [Y/n] ", default='yes')
		project['batch_train_config'] = sfutil.file_input("Location for the batch training TSV config file? [./batch_train.tsv] ",
													root=project['root'], default='./batch_train.tsv', filetype='tsv', verify=False)
		
		if not exists(project['batch_train_config']):
			print("Batch training file not found, creating blank")
			self.create_blank_train_config(project['batch_train_config'])
		
		# Validation strategy
		project['validation_fraction'] = sfutil.float_input("What fraction of training data should be used for validation testing? [0.2] ", valid_range=[0,1], default=0.2)
		project['validation_target'] = sfutil.choice_input("How should validation data be selected by default, per-tile or per-patient? [per-patient] ", valid_choices=['per-tile', 'per-patient'], default='per-patient')
		if project['validation_target'] == 'per-patient':
			project['validation_strategy'] = sfutil.choice_input("Which validation strategy should be used by default, k-fold, bootstrap, or fixed? [k-fold]", valid_choices=['k-fold', 'bootstrap', 'fixed', 'none'], default='k-fold')
		else:
			project['validation_strategy'] = sfutil.choice_input("Which validation strategy should be used by default, k-fold or fixed? ", valid_choices=['k-fold', 'fixed', 'none'])
		if project['validation_strategy'] == 'k-fold':
			project['validation_k_fold'] = sfutil.int_input("What is K? [3] ", default=3)
		elif project['validation_strategy'] == 'bootstrap':
			project['validation_k_fold'] = sfutil.int_input("How many iterations should be performed when bootstrapping? [3] ", default=3)
		else:
			project['validation_k_fold'] = 0

		sfutil.write_json(project, join(project_folder, 'settings.json'))
		self.PROJECT = project

		# Write a sample actions.py file
		with open(join(SOURCE_DIR, 'sample_actions.py'), 'r') as sample_file:
			sample_actions = sample_file.read()
			with open(os.path.join(project_folder, 'actions.py'), 'w') as actions_file:
				actions_file.write(sample_actions)

		print("\nProject configuration saved.\n")
		self.load_project(project_folder)
    
	def evaluate(self, model, outcome_label_headers, hyperparameters=None, filters=None, checkpoint=None,
					eval_k_fold=None, max_tiles_per_slide=0, min_tiles_per_slide=0, normalizer=None,
					normalizer_source=None, input_header=None, permutation_importance=False, histogram=True, 
					save_predictions=False):
		'''Evaluates a saved model on a given set of tfrecords.
		
		Args:
			model:					Path to Tensorflow model to evaluate.
			outcome_label_headers:			Annotation column header that specifies the outcome label.
			hyperparameters:		Path to model's hyperparameters.json file. If None, searches for this file in the same directory as the model.
			filters:				Filters to use when selecting tfrecords on which to perform evaluation.
			checkpoint:				Path to cp.ckpt file to load, if evaluating a saved checkpoint.
			eval_k_fold:			K-fold iteration number to evaluate. If None, will evaluate all tfrecords irrespective of K-fold.
			max_tiles_per_slide:	Will only use up to this many tiles from each slide for evaluation. If zero, will include all tiles.
			min_tiles_per_slide:	Minimum number of tiles a slide must have to be included in evaluation. Default is 0, but
										for best slide-level AUC, a minimum of at least 10 tiles per slide is recommended.
			normalizer:				Normalization strategy to use on image tiles.
			normalizer_source:		Path to normalizer source image.
			permutation_importance:	Bool. True if you want to calculate the permutation feature importance (used to determine relative importance when using multiple model inputs
		'''							
		log.header(f"Evaluating model {sfutil.green(model)}...")
		
		if (input_header is None) and permutation_importance:
			log.warn("Permutation feature importance is designed to be used with multimodal models. Turning off.", 1)
			permutation_importance = False

		manager = multiprocessing.Manager()
		results_dict = manager.dict()
		ctx = multiprocessing.get_context('spawn')
		
		process = ctx.Process(target=project_utils.evaluator, args=(outcome_label_headers, model, self.PROJECT, results_dict, input_header, filters, hyperparameters, 
														checkpoint, eval_k_fold, max_tiles_per_slide, min_tiles_per_slide, normalizer, normalizer_source,
														self.FLAGS, permutation_importance, histogram, save_predictions))
		process.start()
		log.empty(f"Spawning evaluation process (PID: {process.pid})")
		process.join()

		return results_dict

	def extract_dual_tiles(self, tile_um, tile_px, stride_div=1, filters=None, 
							buffer=True, normalizer=None, normalizer_source=None):
		'''Experimental function to extract dual tiles at two different px/um sizes, saving both within the same TFRecord.'''
		import slideflow.slide as sfslide
		import tensorflow as tf

		log.header("Extracting dual-image tiles...")
		extracting_dataset = self.get_dataset(filters=filters, tile_px=tile_px, tile_um=tile_um)

		def extract_tiles_from_slide(slide_path, roi_list, dataset_config, pb):
			root_path = join(dataset_config["tfrecords"], dataset_config["label"])
			if not exists(root_path): 
					os.makedirs(root_path)

			whole_slide = sfslide.SlideReader(slide_path, 
											  tile_px, 
											  tile_um, 
											  stride_div, 
											  roi_list=roi_list, 
											  buffer=buffer, 
											  pb_counter=pb.get_counter(),
											  counter_lock=pb.get_lock(),
											  print_fn=pb.print)

			small_tile_generator = whole_slide.build_generator(dual_extract=True, 
															   normalizer=normalizer, 
															   normalizer_source=normalizer_source)

			tfrecord_name = sfutil.path_to_name(slide_path)
			tfrecord_path = join(root_path, f"{tfrecord_name}.tfrecords")
			records = []

			for image_dict in small_tile_generator():
				label = bytes(tfrecord_name, 'utf-8')
				image_string_dict = {}
				for image_label in image_dict:
					np_image = image_dict[image_label]
					image = Image.fromarray(np_image).convert('RGB')
					with BytesIO() as output:
						image.save(output, format="JPEG")
						image_string = output.getvalue()
						image_string_dict.update({
							image_label: image_string
						})
				records += [[label, image_string_dict]]

			shuffle(records)
			
			with tf.io.TFRecordWriter(tfrecord_path) as writer:
				for label, image_string_dict in records:
					tf_example = sfio.tfrecords.multi_image_example(label, image_string_dict)
					writer.write(tf_example.SerializeToString())

		for dataset_name in self.PROJECT['datasets']:
			log.empty(f"Working on dataset {sfutil.bold(dataset_name)}", 1)
			slide_list = extracting_dataset.get_slide_paths(dataset=dataset_name)
			roi_list = extracting_dataset.get_rois()
			dataset_config = extracting_dataset.datasets[dataset_name]
			log.info(f"Extracting tiles from {len(slide_list)} slides ({tile_um} um, {tile_px} px)", 2)
			pb = ProgressBar(bar_length=5, counter_text='tiles')
			pb.auto_refresh()

			if self.FLAGS['num_threads'] > 1:
				pool = DPool(self.FLAGS['num_threads'])
				pool.map(partial(extract_tiles_from_slide, 
								 roi_list=roi_list, 
								 dataset_config=dataset_config, 
								 pb=pb), 
						 slide_list)
				pool.close()
			else:
				for slide_path in slide_list:
					extract_tiles_from_slide(slide_path, roi_list, dataset_config, pb)
		
		extracting_dataset.update_manifest()

	def tfrecord_report(self, tile_px, tile_um, filters=None, filter_blank=None, dataset=None,
						 destination='auto', normalizer=None, normalizer_source=None):
		'''Creates a PDF report of TFRecords, including 10 example tiles per TFRecord.

		Args:
			tile_px:				Tile width in pixels
			tile_um:				Tile width in microns
			filters:				Dataset filters to use for selecting TFRecords
			filter_blank:			List of label headers; slides that have blank entries in this label header
								 		in the annotations file will be excluded
			destination:			Either 'auto' or explicit filename at which to save the PDF report
			normalizer:				Normalization strategy to use on image tiles
			normalizer_source:		Path to normalizer source image
			dataset:				Name of dataset from which to select TFRecords. If not provided, will use all project datasets
		'''
		from slideflow.slide import ExtractionReport, SlideReport
		import tensorflow as tf

		if dataset: datasets = [dataset] if not isinstance(dataset, list) else dataset
		else:		datasets = self.PROJECT['datasets']

		if normalizer: log.info(f"Using realtime {normalizer} normalization", 1)
		normalizer = None if not normalizer else StainNormalizer(method=normalizer, source=normalizer_source)

		tfrecord_dataset = self.get_dataset(filters=filters, 
											filter_blank=filter_blank, 
											tile_px=tile_px, 
											tile_um=tile_um)
		log.header("Generating TFRecords report...")
		reports = []
		for dataset_name in datasets:
			tfrecord_list = tfrecord_dataset.get_tfrecords(dataset=dataset_name)
			for tfr in tfrecord_list:
				print(f"\r\033[KGenerating report for tfrecord {sfutil.green(sfutil.path_to_name(tfr))}...", end="")
				dataset = tf.data.TFRecordDataset(tfr)
				parser = sfio.tfrecords.get_tfrecord_parser(tfr, ("image_raw"), to_numpy=True, decode_images=False)
				sample_tiles = []
				for i, record in enumerate(dataset):
					if i > 9: break
					image_raw_data = parser(record)

					if normalizer:
						image_raw_data = normalizer.jpeg_to_jpeg(image_raw_data)

					sample_tiles += [image_raw_data]
				reports += [SlideReport(sample_tiles, tfr)]
		print("\r\033[K", end="")
		log.empty("Generating PDF (this may take some time)...", 1)
		pdf_report = ExtractionReport(reports, tile_px=tile_px, tile_um=tile_um)
		timestring = datetime.now().strftime("%Y%m%d-%H%M%S")
		filename = destination if destination != 'auto' else join(self.PROJECT['root'], f'tfrecord_report-{timestring}.pdf')
		pdf_report.save(filename)
		log.complete(f"TFRecord report saved to {sfutil.green(filename)}", 1)

	def slide_report(self, tile_px, tile_um, filters=None, filter_blank=None, dataset=None, 
						stride_div=1, destination='auto', tma=False, enable_downsample=False, 
						roi_method='inside', skip_missing_roi=True, normalizer=None, normalizer_source=None):
		'''Creates a PDF report of slides, including images of 10 example extracted tiles.

		Args:
			tile_px:				Tile width in pixels
			tile_um:				Tile width in microns
			filters:				Dataset filters to use for selecting TFRecords
			filter_blank:			List of label headers; slides that have blank entries in this label header
								 		in the annotations file will be excluded
			dataset:				Name of dataset from which to select TFRecords. If not provided, will use all project datasets
			stride_div:				Stride divisor for tile extraction
			destination:			Either 'auto' or explicit filename at which to save the PDF report
			tma:					Bool, if True, interprets slides to be TMA (tumor microarrays)
			enable_downsample:		Bool, if True, enables downsampling during tile extraction
			roi_method:				Either 'inside', 'outside', or 'ignore'. Determines how ROIs will guide tile extraction
			skip_missing_roi:		Bool, if True, will skip tiles that are missing ROIs
			normalizer:				Normalization strategy to use on image tiles
			normalizer_source:		Path to normalizer source image
		'''
		import slideflow.slide as sfslide

		if dataset: datasets = [dataset] if not isinstance(dataset, list) else dataset
		else:		datasets = self.PROJECT['datasets']

		extracting_dataset = self.get_dataset(filters=filters, 
											  filter_blank=filter_blank, 
											  tile_px=tile_px, 
											  tile_um=tile_um)

		log.header("Generating slide report...")
		reports = []
		for dataset_name in datasets:
			roi_dir = extracting_dataset.datasets[dataset_name]['roi']
			slide_list = extracting_dataset.get_slide_paths(dataset=dataset_name)

			# Function to extract tiles from a slide
			def get_slide_report(slide_path):
				print(f"\r\033[KGenerating report for slide {sfutil.green(sfutil.path_to_name(slide_path))}...", end="")

				if tma:
					whole_slide = sfslide.TMAReader(slide_path, 
													tile_px, 
													tile_um, 
													stride_div, 
													enable_downsample=enable_downsample, 
													silent=True)
				else:
					whole_slide = sfslide.SlideReader(slide_path, 
													  tile_px, 
													  tile_um, 
													  stride_div, 
													  enable_downsample=enable_downsample, 
													  roi_dir=roi_dir,
													  roi_method=roi_method,
													  silent=True,
													  skip_missing_roi=skip_missing_roi)

				if not whole_slide.loaded_correctly():
					return

				report = whole_slide.extract_tiles(normalizer=normalizer, normalizer_source=normalizer_source)
				return report

			for slide_path in slide_list:
				report = get_slide_report(slide_path)
				reports += [report]
		print("\r\033[K", end="")
		log.empty("Generating PDF (this may take some time)...", )
		pdf_report = sfslide.ExtractionReport(reports, tile_px=tile_px, tile_um=tile_um)
		timestring = datetime.now().strftime("%Y%m%d-%H%M%S")
		filename = destination if destination != 'auto' else join(self.PROJECT['root'], f'tile_extraction_report-{timestring}.pdf')
		pdf_report.save(filename)
		log.complete(f"Slide report saved to {sfutil.green(filename)}", 1)

	def predict_wsi(self, model_path, tile_px, tile_um, export_dir, filters=None, filter_blank=None, stride_div=1, 
						enable_downsample=False, roi_method='inside', skip_missing_roi=False, 
						dataset=None, normalizer=None, normalizer_source=None, 
						whitespace_fraction=1.0, whitespace_threshold=230, grayspace_fraction=0.6, 
						grayspace_threshold=0.05, randomize_origin=False, buffer=None, num_threads=-1):

		import slideflow.slide as sfslide

		log.header("Generating WSI prediction / activation maps...")
		if not exists(export_dir):
			os.makedirs(export_dir)
		if dataset: datasets = [dataset] if not isinstance(dataset, list) else dataset
		else:		datasets = self.PROJECT['datasets']

		# Load dataset for evaluation
		extracting_dataset = self.get_dataset(filters=filters, 
											  filter_blank=filter_blank, 
											  tile_px=tile_px, 
											  tile_um=tile_um, 
											  verification='slides')
		# Info logging
		if normalizer: log.info(f"Using {sfutil.bold(normalizer)} normalization", 1)
		if whitespace_fraction < 1: log.info(f"Filtering tiles by whitespace fraction (exclude if >={whitespace_fraction*100:.0f}% whitespace, whitespace = RGB avg > {whitespace_threshold})", 1)

		for dataset_name in datasets:
			log.empty(f"Working on dataset {sfutil.bold(dataset_name)}", 1)
			roi_dir = extracting_dataset.datasets[dataset_name]['roi']
			dataset_config = extracting_dataset.datasets[dataset_name]

			# Prepare list of slides for extraction
			slide_list = extracting_dataset.get_slide_paths(dataset=dataset_name)
			log.info(f"Generating predictions for {len(slide_list)} slides ({tile_um} um, {tile_px} px)", 1)

			# Verify slides and estimate total number of tiles
			log.empty("Verifying slides...", 1)
			total_tiles = 0
			for slide_path in slide_list:
				slide = sfslide.SlideReader(slide_path, 
											tile_px, 
											tile_um, 
											stride_div, 
											roi_dir=roi_dir,
											roi_method=roi_method,
											skip_missing_roi=False,
											silent=True,
											buffer=None)
				print(f"\r\033[KVerified {sfutil.green(slide.name)} (approx. {slide.estimated_num_tiles} tiles)", end="")
				total_tiles += slide.estimated_num_tiles
				del(slide)
			if log.INFO_LEVEL > 0: print("\r\033[K", end='')
			log.complete(f"Verification complete. Total estimated tiles to extract: {total_tiles}", 1)
			
			if total_tiles:
				pb = ProgressBar(total_tiles, 
								counter_text='tiles', 
								leadtext="Extracting tiles... ", 
								show_counter=True, 
								show_eta=True)
				pb.auto_refresh()
				pb_counter = pb.get_counter()
				pb_lock = pb.get_lock()
				print_fn = pb.print
			else:
				pb_counter, pb_lock, print_fn = None

			# Function to extract tiles from a slide
			def predict_wsi(slide_path, downsample):
				print_func = print if not pb else pb.print
				log.empty(f"Working on slide {sfutil.path_to_name(slide_path)}", 1, print_func)
				whole_slide = sfslide.SlideReader(slide_path,
													tile_px,
													tile_um,
													stride_div,
													enable_downsample=downsample, 
													roi_dir=roi_dir,
													roi_method=roi_method,
													randomize_origin=randomize_origin,
													skip_missing_roi=skip_missing_roi,
													buffer=buffer,
													pb_counter=pb_counter,
													counter_lock=pb_lock,
													print_fn=print_fn)

				if not whole_slide.loaded_correctly():
					return

				try:
					wsi_grid = whole_slide.predict(model=model_path,
												   normalizer=normalizer,
												   normalizer_source=normalizer_source,
												   whitespace_fraction=whitespace_fraction,
												   whitespace_threshold=whitespace_threshold,
												   grayspace_fraction=grayspace_fraction,
												   grayspace_threshold=grayspace_threshold)

					with open (join(export_dir, whole_slide.name+".pkl"), 'wb') as pkl_file:
						pickle.dump(wsi_grid, pkl_file)

				except sfslide.TileCorruptionError:
					if downsample:
						log.warn(f"Corrupt tile in {sfutil.green(sfutil.path_to_name(slide_path))}; will try re-extraction with downsampling disabled", 1, print_func)
						predict_wsi(slide_path, downsample=False)
					else:
						log.error(f"Corrupt tile in {sfutil.green(sfutil.path_to_name(slide_path))}; skipping slide", 1, print_func)
						return None

			# Use multithreading if specified, extracting tiles from all slides in the filtered list
			if num_threads == -1: num_threads = self.FLAGS['num_threads']
			if num_threads > 1 and len(slide_list):
				q = queue.Queue()
				task_finished = False
				
				def worker():
					while True:
						try:
							path = q.get()
							if buffer and buffer != 'vmtouch':
								buffered_path = join(buffer, os.path.basename(path))
								predict_wsi(buffered_path, enable_downsample)
								os.remove(buffered_path)
							else:
								predict_wsi(path, enable_downsample)
							q.task_done()
						except queue.Empty:
							if task_finished:
								return

				threads = [threading.Thread(target=worker, daemon=True) for t in range(num_threads)]
				for thread in threads:
					thread.start()

				for slide_path in slide_list:
					if buffer and buffer != 'vmtouch':
						while True:
							try:
								shutil.copyfile(slide_path, join(buffer, os.path.basename(slide_path)))
								q.put(slide_path)
								break
							except OSError:
								time.sleep(5)
					else:
						q.put(slide_path)
				q.join()
				task_finished = True
				if pb: pb.end()
			else:
				for slide_path in slide_list:
					predict_wsi(slide_path, enable_downsample)
				if pb: pb.end()

	def extract_tiles(self, tile_px, tile_um, filters=None, filter_blank=None, stride_div=1, 
						tma=False, full_core=False, save_tiles=False, save_tfrecord=True,
						enable_downsample=False, roi_method='inside', skip_missing_roi=True, 
						skip_extracted=True, dataset=None, normalizer=None, normalizer_source=None, 
						whitespace_fraction=1.0, whitespace_threshold=230, grayspace_fraction=0.6, 
						grayspace_threshold=0.05, img_format='png', randomize_origin=False, buffer=None, shuffle=True,
						num_workers=4, threads_per_worker=4):
		'''Extract tiles from a group of slides; save a percentage of tiles for validation testing if the 
		validation target is 'per-patient'; and generate TFRecord files from the raw images.
		
		Args:
			tile_px:				Tile size in pixels.
			tile_um:				Tile size in microns.
			filters:				Dataset filters to use when selecting slides for tile extraction.
			stride_div:				Stride divisor to use when extracting tiles. A stride of 1 will extract non-overlapping tiles. 
										A stride_div of 2 will extract overlapping tiles, with a stride equal to 50% of the tile width.
			tma:					Bool. If True, reads slides as Tumor Micro-Arrays (TMAs), detecting and extracting tumor cores.
			full_core:				Bool. Only used if extracting from TMA. If True, will save entire TMA core as image. Otherwise, will extract sub-images
										from each core using the given tile micron size.
			save_tiles:				Bool. If True, will save JPEG images of extracted tiles to corresponding tile directory.
			save_tfrecord:			Bool. If True, will save JPEG-compressed image data from extracted tiles into TFRecords in the corresponding TFRecord directory.
			enable_downsample:		Bool. If True, enables the use of downsampling while reading slide images. This may result in corrupted image tiles
										if downsampled slide layers are corrupted or not fully generated. Manual confirmation of tile integrity is recommended.
			roi_method:				Either 'inside', 'outside', or 'ignore'. Whether to extract tiles inside or outside the ROIs.
			skip_missing_roi:		Bool. If True, will skip slides that are missing ROIs
			skip_extracted:			Bool. If True, will skip slides that have already been fully extracted
			dataset:				Name of dataset from which to select slides for extraction. If not provided, will default to all datasets in project
			normalizer:				Normalization strategy to use on image tiles
			normalizer_source:		Path to normalizer source image
			whitespace_fraction:	Float 0-1. Fraction of whitespace which causes a tile to be discarded. If 1, will not perform whitespace filtering.
			whitespace_threshold:	Int 0-255. Threshold above which a pixel (averaged across R,G,B) is considered whitespace.
			grayspace_fraction:		Float 0-1. Fraction of grayspace which causes a tile to be discarded. If 1, will not perform grayspace filtering.
			grayspace_threshold:	Int 0-1. HSV (hue, saturation, value) is calculated for each pixel. If a pixel's saturation is below this threshold, it is considered grayspace.
			buffer:					Either 'vmtouch' or path to directory. If vmtouch, will use vmtouch to preload slide into memory before extraction.
										If a directory, slides will be copied to the directory as a buffer before extraction.
										Either method vastly improves tile extraction for slides on HDDs by maximizing sequential read speed
			num_workers:			Number of slides from which to be extracting tiles simultaneously.
			threads_per_worker:		Number of processes to allocate to each slide for tile extraction.
		'''

		import slideflow.slide as sfslide

		log.header("Extracting image tiles...")

		if not save_tiles and not save_tfrecord:
			log.error("Either save_tiles or save_tfrecord must be true to extract tiles.", 1)
			return
		
		if dataset: datasets = [dataset] if not isinstance(dataset, list) else dataset
		else:		datasets = self.PROJECT['datasets']

		# Load dataset for evaluation
		extracting_dataset = self.get_dataset(filters=filters, 
											  filter_blank=filter_blank, 
											  tile_px=tile_px, 
											  tile_um=tile_um, 
											  verification='slides')

		# Prepare validation/training subsets if per-tile validation is being used
		if self.PROJECT['validation_target'] == 'per-tile':
			if self.PROJECT['validation_strategy'] == 'boostrap':
				log.warn("Validation bootstrapping is not supported when the validation target is per-tile; will generate random fixed validation target", 1)
			if self.PROJECT['validation_strategy'] in ('bootstrap', 'fixed'):
				# Split the extracted tiles into two groups
				split_fraction = [-1, self.PROJECT['validation_fraction']]
				split_names = ['training', 'validation']
			if self.PROJECT['validation_strategy'] == 'k-fold':
				split_fraction = [-1] * self.PROJECT['validation_k_fold']
				split_names = [f'kfold-{i}' for i in range(self.PROJECT['validation_k_fold'])]
		else:
			split, split_fraction, split_names = None, None, None

		if normalizer: log.info(f"Extracting tiles using {sfutil.bold(normalizer)} normalization", 1)
		if whitespace_fraction < 1: log.info(f"Filtering tiles by whitespace fraction (exclude if >={whitespace_fraction*100:.0f}% whitespace, whitespace = RGB avg > {whitespace_threshold})", 1)

		for dataset_name in datasets:
			log.empty(f"Working on dataset {sfutil.bold(dataset_name)}", 1)

			tiles_dir = join(extracting_dataset.datasets[dataset_name]['tiles'], 
								extracting_dataset.datasets[dataset_name]['label'])
			roi_dir = extracting_dataset.datasets[dataset_name]['roi']
			dataset_config = extracting_dataset.datasets[dataset_name]
			tfrecord_dir = join(dataset_config["tfrecords"], dataset_config["label"])
			if save_tfrecord and not exists(tfrecord_dir):
				os.makedirs(tfrecord_dir)
			if save_tiles and not os.path.exists(tiles_dir):
				os.makedirs(tiles_dir)

			# Prepare list of slides for extraction
			slide_list = extracting_dataset.get_slide_paths(dataset=dataset_name)
			
			# Check for interrupted or already-extracted tfrecords
			if skip_extracted and save_tfrecord:
				already_extracted_tfrecords = [sfutil.path_to_name(tfr) for tfr in extracting_dataset.get_tfrecords(dataset=dataset_name)]
				interrupted = [sfutil.path_to_name(marker) for marker in glob(join((tfrecord_dir if tfrecord_dir else tiles_dir), '*.unfinished'))]
				if len(interrupted):
					log.info(f'Interrupted tile extraction detected in {len(interrupted)} tfrecords, will re-extract these slides', 1)
					for interrupted_slide in interrupted:
						log.empty(interrupted_slide, 2)
						if interrupted_slide in already_extracted_tfrecords:
							del(already_extracted_tfrecords[already_extracted_tfrecords.index(interrupted_slide)])
					
				slide_list = [slide for slide in slide_list if sfutil.path_to_name(slide) not in already_extracted_tfrecords]
				if len(already_extracted_tfrecords):
					log.info(f"Skipping tile extraction for {len(already_extracted_tfrecords)} slides; TFRecords already generated.", 1)	
			log.info(f"Extracting tiles from {len(slide_list)} slides ({tile_um} um, {tile_px} px)", 1)

			# Verify slides and estimate total number of tiles
			log.empty("Verifying slides...", 1)
			total_tiles = 0
			for slide_path in slide_list:
				if tma:
					slide = sfslide.TMAReader(slide_path, tile_px, tile_um, stride_div, silent=True)
				else:
					slide = sfslide.SlideReader(slide_path, 
												tile_px, 
												tile_um, 
												stride_div, 
												roi_dir=roi_dir,
												roi_method=roi_method,
												skip_missing_roi=False,
												silent=True)
				print(f"\r\033[KVerified {sfutil.green(slide.name)} (approx. {slide.estimated_num_tiles} tiles)", end="")
				total_tiles += slide.estimated_num_tiles
				del(slide)
			if log.INFO_LEVEL > 0: print("\r\033[K", end='')
			log.complete(f"Verification complete. Total estimated tiles to extract: {total_tiles}", 1)
			
			# Use multithreading if specified, extracting tiles from all slides in the filtered list
			if len(slide_list):
				q = queue.Queue()
				task_finished = False
				manager = multiprocessing.Manager()
				ctx = multiprocessing.get_context('spawn')
				reports = manager.dict()
				counter = manager.Value('i', 0)
				counter_lock = manager.Lock()

				if total_tiles:
					pb = ProgressBar(total_tiles, counter_text='tiles', leadtext="Extracting tiles... ", show_counter=True, show_eta=True, mp_counter=counter, mp_lock=counter_lock)
					pb.auto_refresh()
				else:
					pb = None

				# Worker to grab slide path from queue and start tile extraction
				def worker():
					while True:
						try:
							path = q.get()
							process = ctx.Process(target=project_utils.tile_extractor, args=(path, roi_dir, roi_method, skip_missing_roi, randomize_origin,
																				img_format, tma, full_core, shuffle, tile_px, tile_um, stride_div, False,
																				whitespace_fraction, whitespace_threshold, grayspace_fraction, grayspace_threshold, normalizer, 
																				normalizer_source, split_fraction, split_names, self.PROJECT['root'], tfrecord_dir, tiles_dir, 
																				save_tiles, save_tfrecord, buffer, threads_per_worker, counter, counter_lock))

							process.start()
							process.join()
							if buffer and buffer != 'vmtouch':
								os.remove(path)
							q.task_done()
						except queue.Empty:
							if task_finished:
								return

				# Start the worker threads
				threads = [threading.Thread(target=worker, daemon=True) for t in range(num_workers)]
				for thread in threads:
					thread.start()

				# Put each slide path into queue
				for slide_path in slide_list:
					warned = False
					if buffer and buffer != 'vmtouch':
						while True:
							if q.qsize() < num_workers:
								try:
									buffered_path = join(buffer, os.path.basename(slide_path))
									shutil.copy(slide_path, buffered_path)
									q.put(buffered_path)
									break
								except OSError as e:
									if not warned:
										log.warn(f"OSError encountered for slide {sfutil._shortname(sfutil.path_to_name(slide_path))}: buffer likely full")
										log.info(f"Q size: {q.qsize()}")
										warned = True
									time.sleep(1)
							else:
								time.sleep(1)
					else:
						q.put(slide_path)
				q.join()
				task_finished = True
				if pb: pb.end()
				log.empty("Generating PDF (this may take some time)...", )
				pdf_report = sfslide.ExtractionReport(reports.values(), tile_px=tile_px, tile_um=tile_um)
				timestring = datetime.now().strftime("%Y%m%d-%H%M%S")
				pdf_report.save(join(self.PROJECT['root'], f'tile_extraction_report-{timestring}.pdf'))

			# Update manifest
			extracting_dataset.update_manifest()

	def generate_activations(self, model, outcome_label_headers=None, layers=['postconv'], filters=None, filter_blank=None, 
								focus_nodes=[], node_exclusion=False, activations_export=None,
								activations_cache=None, normalizer=None, normalizer_source=None, 
								max_tiles_per_slide=0, min_tiles_per_slide=None, model_format=None, include_logits=True,
								batch_size=None, torch_export=None, isolated_thread=False):
		'''Calculates final layer activations and displays information regarding the most significant final layer nodes.
		Note: GPU memory will remain in use, as the Keras model associated with the visualizer is active.
		
		Args:
			model:				Path to Tensorflow model
			outcome_label_headers:		Column header in annotations file; used for category-level comparisons
			filters:			Dataset filters for selecting TFRecords
			filter_blank:		List of label headers; slides that have blank entries in this label header
									in the annotations file will be excluded
			focus_nodes:		List of int, indicates which nodes are of interest for subsequent analysis
			activations_export:	Path to CSV file, if provided, will save activations in CSV format to this file
			activations_cache:	Either 'default' or path to 'PKL' file; will save activations to this file in PKL format as cache
			normalizer:			Normalization strategy to use on image tiles
			normalizer_source:	Path to normalizer source image
			model_format:		Optional. May supply format of saved Slideflow Keras model if the model was made with a legacy version.
									Default value will be slideflow.model.MODEL_FORMAT_CURRENT,
									but slideflow.model.MODEL_FORMAT_LEGACY may be supplied.
			batch_size:			Batch size to use when calculating activations.
		'''

		if isolated_thread:
			manager = multiprocessing.Manager()
			results_dict = manager.dict()
			ctx = multiprocessing.get_context('spawn')
			
			process = ctx.Process(target=project_utils.activations_generator, args=(self.PROJECT, model, outcome_label_headers, layers, filters, filter_blank, 
																		focus_nodes, node_exclusion, activations_export,
																		activations_cache, normalizer, normalizer_source, 
																		max_tiles_per_slide, min_tiles_per_slide, model_format, 
																		include_logits, batch_size, torch_export, results_dict))
			process.start()
			log.empty(f"Spawning activations process (PID: {process.pid})")
			process.join()
			return results_dict
		else:
			AV = project_utils.activations_generator(self.PROJECT, model, outcome_label_headers, layers, filters, filter_blank, 
										focus_nodes, node_exclusion, activations_export,
										activations_cache, normalizer, normalizer_source, 
										max_tiles_per_slide, min_tiles_per_slide, model_format, 
										include_logits, batch_size, torch_export, None)
			return AV

	def generate_heatmaps(self, model, filters=None, filter_blank=None, directory=None, resolution='low', 
							interpolation='none', show_roi=True, roi_method='inside', logit_cmap=None, skip_thumb=False, 
							normalizer=None, normalizer_source=None, buffer=True, isolated_thread=True, 
							num_threads='auto', model_format=None):
		'''Creates predictive heatmap overlays on a set of slides. 

		Args:
			model:				Path to Tensorflow model with which predictions will be generated.
			filters:			Dataset filters to use when selecting slides for which to generate heatmaps.
			filter_blank:		List of label headers; slides that have blank entries in this label header
								 	in the annotations file will be excluded
			directory:			Directory in which to save heatmap images.
			resolution:			Heatmap resolution (determines stride of tile predictions). 
									"low" uses a stride equal to tile width.
									"medium" uses a stride equal 1/2 tile width.
									"high" uses a stride equal to 1/4 tile width.
			interpolation:		Interpolation strategy for smoothing heatmap predictions (matplotlib imshow interpolation options). 
			show_roi:			Bool. If True, will show ROI on heatmaps.
			roi_method:			'inside', 'outside', or 'none'. Determines where heatmap should be made with respect to annotated ROI.
			logit_cmap:			Either a function or a dictionary used to create heatmap colormap.
									If None (default), separate heatmaps will be generated for each label category, with color representing likelihood of category prediction.
									Each image tile will generate a list of predictions of length O, 
									where O is the number of label categories.
									If logit_cmap is a function, then this logit prediction list will be passed to the function,
									and the function is expected to return [R, G, B] values which will be displayed. isolated_thread must be true if a function is passed.
									If the logit_cmap is a dictionary, it should map 'r', 'g', and 'b' to label indices;
									The prediction for these label categories will be mapped to the corresponding colors.
									Thus, the corresponding color will only reflect predictions of up to three label categories.
										Example (this would map prediction for label 0 to the red colorspace, label 3 to green colorspace, etc):
										{'r': 0, 'g': 3, 'b': 1 }
			skip_thumb:			Bool. If True, will not display thumbnail with heatmap.
			normalizer:			Normalization strategy to use on image tiles
			normalizer_source:	Path to normalizer source image
			buffer:				Either 'vmtouch' or path to directory. If vmtouch, will use vmtouch to preload slide into memory before extraction.
									If a directory, slides will be copied to the directory as a buffer before extraction.
									Either method vastly improves tile extraction for slides on HDDs by maximizing sequential read speed
			isolated_thread:	Bool. If True, will wrap function in separate process, allowing GPU memory to be freed after completion.
									If False, will perform as single thread (GPU memory may not be freed after completion). 
									Allows use for functions being passed to logit_cmap (functions are not pickleable).
									
			model_format:		Optional. May supply format of saved Slideflow Keras model if the model was made with a legacy version.
									Default value will be slideflow.model.MODEL_FORMAT_CURRENT,
									but slideflow.model.MODEL_FORMAT_LEGACY may be supplied.
		'''
		log.header("Generating heatmaps...")

		# Prepare dataset
		hp_data = sfutil.load_json(join(dirname(model), 'hyperparameters.json'))
		heatmaps_dataset = self.get_dataset(filters=filters,
											filter_blank=filter_blank,
											tile_px=hp_data['hp']['tile_px'],
											tile_um=hp_data['hp']['tile_um'])
		slide_list = heatmaps_dataset.get_slide_paths()
		roi_list = heatmaps_dataset.get_rois()

		# Attempt to auto-detect supplied model name
		detected_model_name = sfutil.path_to_name(model)
		hp_file = join(*model.split('/')[:-1], 'hyperparameters.json')
		if exists(hp_file):
			loaded_hp = sfutil.load_json(hp_file)
			if 'model_name' in loaded_hp:
				detected_model_name = loaded_hp['model_name']
		
		# Make output directory
		heatmaps_folder = directory if directory else os.path.join(self.PROJECT['root'], 'heatmaps', detected_model_name)
		if not exists(heatmaps_folder): os.makedirs(heatmaps_folder)

		# Heatmap processes
		ctx = multiprocessing.get_context('spawn')
		for slide in slide_list:
			if isolated_thread:
				process = ctx.Process(target=project_utils.heatmap_generator, args=(slide, model, heatmaps_folder, roi_list, show_roi, roi_method,
																		resolution, interpolation, self.PROJECT, logit_cmap, skip_thumb,
																		buffer, normalizer, normalizer_source, model_format, num_threads, self.FLAGS))
				process.start()
				log.empty(f"Spawning heatmaps process (PID: {process.pid})")
				process.join()
			else:
				project_utils.heatmap_generator(slide, model, heatmaps_folder, roi_list, show_roi, roi_method,
									resolution, interpolation, self.PROJECT, logit_cmap, skip_thumb,
									buffer, normalizer, normalizer_source, model_format, num_threads, self.FLAGS)

	def generate_mosaic(self, model, mosaic_filename=None, umap_filename=None, outcome_label_headers=None, filters=None,
						filter_blank=None, focus_filters=None, resolution="low", num_tiles_x=50, 
						max_tiles_per_slide=100, expanded=False, map_slide=None, show_prediction=None, 
						restrict_prediction=None, predict_on_axes=None, whitespace_on_axes=False, 
						label_names=None, cmap=None, model_type=None, umap_cache='default', 
						activations_cache='default', activations_export=None, umap_export=None, use_float=False, 
						normalizer=None, normalizer_source=None, low_memory=False, model_format=None):

		'''Generates a mosaic map by overlaying images onto a set of mapped tiles.
			Image tiles are extracted from the provided set of TFRecords, and predictions + post-convolutional node activations are calculated using the specified model.
			Tiles are mapped either with dimensionality reduction on post-convolutional layer activations (default behavior), 
			or by using outcome predictions for two categories, mapped to X- and Y-axis (via predict_on_axes).
		
		Args:
			model:					Path to Tensorflow model to use when generating layer activations.
			mosaic_filename:		Filename for mosaic image. If not provided, mosaic will not be calculated or saved. Will be saved in project mosaic directory.
			umap_filename:			Filename for UMAP plot image. If not provided, plot will not be saved. Will be saved in project stats directory.
			outcome_label_headers:			Column name in annotations file from which to read category labels.
			filters:				Dataset filters to use when selecting slides to include the mosaic.
			filter_blank:			List of label headers; slides that have blank entries in this label header
								 		in the annotations file will be excluded
			focus_filters:			Dataset filters to use when selecting slides to highlight on the mosaic.
			resolution:				Resolution of the mosaic map. Impacts size of the final figure. Either low, medium, or high.
			num_tiles_x:			Specifies the size of the mosaic map grid.
			max_tiles_per_slide:	Limits the number of tiles taken from each slide. Too high of a number may introduce memory issues.
			expanded:				Bool. If False, will limit tile assignment to the corresponding grid space (strict display).
										If True, allows for display of nearby tiles if a given grid is empty.
			map_slide:				None (default), 'centroid', or 'average'. If provided, will map slides using slide-level calculations, either mapping centroid tiles if 'centroid',
										or calculating node averages across all tiles in a slide and mapping slide-level node averages, if 'average'
			show_prediction:		May be either int or string, corresponding to label category. Predictions for this category will be displayed
										On the exported UMAP plot.
			restrict_prediction:	List of int, if provided, will restrict predictions to only these categories
										(final tile-level prediction is made by choosing category with highest logit)
			predict_on_axes:		(int, int). Each int corresponds to an label category id. 
										If provided, predictions are generated for these two labels categories;
										tiles are then mapped with these predictions with the pattern (x, y)
										and the mosaic is generated from this map. This replaces the default
										dimensionality reduction mapping.
			label_names:			Dict mapping label id (int) to string names. Saved in the hyperparameters file as "outcome_labels"
			cmap:					Colormap mapping labels to colors for display on UMAP plot
			model_type:				Indicates label type. May be 'categorical', 'linear', or 'cph' (Cox Proportional Hazards)
			umap_cache:				Either 'default' or path to PKL file in which to save/cache UMAP coordinates
			activations_cache:		Either 'default' or path to PKL file in which to save/cache nodal activations
			activations_export:		Filename for CSV export of activations. Will be saved in project stats directory.
			umap_export:			Filename for CSV export of UMAP coordinates. Will be saved in project stats directory.
			use_float:				Bool, if True, assumes labels are float / linear (as opposed to categorical)
			normalizer:				Normalization strategy to use on image tiles
			normalizer_source:		Path to normalizer source image
			low_memory:				Bool, if True, will attempt to limit memory during UMAP calculations at the cost of increased computational complexity
			model_format:			Optional. May supply format of saved Slideflow Keras model if the model was made with a legacy version.
										Default value will be slideflow.model.MODEL_FORMAT_CURRENT,
										but slideflow.model.MODEL_FORMAT_LEGACY may be supplied.
		'''
		from slideflow.activations import ActivationsVisualizer
		from slideflow.mosaic import Mosaic

		log.header("Generating mosaic map...")

		# Set up paths
		stats_root = join(self.PROJECT['root'], 'stats')
		mosaic_root = join(self.PROJECT['root'], 'mosaic')
		if not exists(stats_root): os.makedirs(stats_root)
		if not exists(mosaic_root): os.makedirs(mosaic_root)
		if umap_cache and umap_cache == 'default':
			umap_cache = join(stats_root, 'umap_cache.pkl')
		elif umap_cache:
			umap_cache = join(stats_root, umap_cache)

		# Prepare dataset & model
		hp_data = sfutil.load_json(join(dirname(model), 'hyperparameters.json'))
		mosaic_dataset = self.get_dataset(filters=filters,
										  filter_blank=filter_blank,
										  tile_px=hp_data['hp']['tile_px'],
										  tile_um=hp_data['hp']['tile_um'])
		tfrecords_list = mosaic_dataset.get_tfrecords()
		if focus_filters:
			mosaic_dataset.apply_filters(focus_filters)
			focus_list = mosaic_dataset.get_tfrecords()
		else:
			focus_list = None
		log.info(f"Generating mosaic from {len(tfrecords_list)} slides, with focus on {0 if not focus_list else len(focus_list)} slides.", 1)

		# If a header category is supplied and we are not showing predictions, then assign slide labels from annotations
		if model_type == 'linear': use_float = True
		if outcome_label_headers and (show_prediction is None):
			slide_labels = mosaic_dataset.slide_to_label(outcome_label_headers, use_float=use_float)
		else:
			slide_labels = {}

		# If showing predictions, try to automatically load prediction labels
		if (show_prediction is not None) and (not use_float) and (not label_names):
			if exists(join(dirname(model), 'hyperparameters.json')):
				model_hyperparameters = sfutil.load_json(join(dirname(model), 'hyperparameters.json'))
				outcome_labels = model_hyperparameters['outcome_labels']
				model_type = model_type if model_type else model_hyperparameters['model_type']
				log.info(f'Automatically loaded prediction labels found at {sfutil.green(dirname(model))}', 1)
			else:
				log.info(f'Unable to auto-detect prediction labels from model hyperparameters file', 1)
				
		# Initialize mosaic, umap, and ActivationsVisualizer
		mosaic, umap = None, None

		AV = ActivationsVisualizer(model=model,
								   tfrecords=tfrecords_list, 
								   root_dir=self.PROJECT['root'],
								   image_size=hp_data['hp']['tile_px'],
								   focus_nodes=None,
								   use_fp16=self.PROJECT['use_fp16'],
								   normalizer=normalizer,
								   normalizer_source=normalizer_source,
								   batch_size=self.FLAGS['eval_batch_size'],
								   activations_export=None if not activations_export else join(stats_root, activations_export),
								   max_tiles_per_slide=max_tiles_per_slide,
								   activations_cache=activations_cache,
								   manifest=mosaic_dataset.get_manifest(),
								   model_format=model_format)

		if predict_on_axes:
			# Create mosaic using x- and y- axis corresponding to label predictions
			umap_x, umap_y, umap_meta = AV.map_to_predictions(predict_on_axes[0], predict_on_axes[1])
			umap = TFRecordMap.from_precalculated(tfrecords=mosaic_dataset.get_tfrecords(),
												  slides=mosaic_dataset.get_slides(),
												  x=umap_x,
												  y=umap_y,
												  meta=umap_meta)
		elif whitespace_on_axes:
			umap_x, umap_y, umap_meta = AV.map_to_whitespace(whitespace_threshold=230)
			umap = TFRecordMap.from_precalculated(tfrecords=mosaic_dataset.get_tfrecords(),
												  slides=mosaic_dataset.get_slides(),
												  x=umap_x,
												  y=umap_y,
												  meta=umap_meta)
		else:
			# Create mosaic map from dimensionality reduction on post-convolutional layer activations
			umap = TFRecordMap.from_activations(AV, 
												map_slide=map_slide,
												prediction_filter=restrict_prediction,
												cache=umap_cache,
												low_memory=low_memory,
												max_tiles_per_slide=max_tiles_per_slide)

		# If displaying centroid AND predictions, then show slide-level predictions rather than tile-level predictions
		if (map_slide=='centroid') and show_prediction is not None:
			log.info("Showing slide-level predictions at point of centroid", 1)

			# If not model has not been assigned, assume categorical model
			model_type = model_type if model_type else 'categorical'

			# Get predictions
			if model_type == 'categorical':
				slide_predictions, slide_percentages = AV.get_slide_level_categorical_predictions(prediction_filter=restrict_prediction)
			else:
				slide_predictions = slide_percentages = AV.get_slide_level_linear_predictions()

			# If show_prediction is provided (either a number or string), then display ONLY the prediction for the provided category, as a colormap
			if type(show_prediction) == int:
				log.info(f"Showing prediction for label {show_prediction} as colormap", 1)
				slide_labels = {k:v[show_prediction] for k, v in slide_percentages.items()}
				show_prediction = None
				use_float = True
			elif type(show_prediction) == str:
				log.info(f"Showing prediction for label {show_prediction} as colormap", 1)
				reversed_labels = {v:k for k, v in outcome_labels.items()}
				if show_prediction not in reversed_labels:
					raise ValueError(f"Unknown label category `{show_prediction}`")
				slide_labels = {k:v[int(reversed_labels[show_prediction])] for k, v in slide_percentages.items()}
				show_prediction = None
				use_float = True
			elif use_float:
				# Displaying linear predictions needs to be implemented here
				raise TypeError("If showing prediction and use_float is True, please pass desired label category for prediction to `show_prediction`.")
			# Otherwise, show_prediction is assumed to be just "True", in which case show categorical predictions
			else:
				try:
					slide_labels = {k:outcome_labels[v] for (k,v) in slide_predictions.items()}
				except KeyError:
					# Try interpreting prediction label keys as strings
					slide_labels = {k:outcome_labels[str(v)] for (k,v) in slide_predictions.items()}

		if umap_filename:
			if slide_labels:
				umap.label_by_slide(slide_labels)
			if show_prediction and (map_slide != 'centroid'):
				umap.label_by_tile_meta('prediction', translation_dict=outcome_labels)
			umap.filter(mosaic_dataset.get_slides())
			umap.save_2d_plot(join(stats_root, umap_filename), cmap=cmap, use_float=use_float)
		if umap_export:
			umap.export_to_csv(join(stats_root, umap_export))

		if mosaic_filename:
			mosaic = Mosaic(umap, 
							leniency=1.5,
							expanded=expanded,
							tile_zoom=15,
							num_tiles_x=num_tiles_x,
							resolution=resolution,
							normalizer=normalizer,
							normalizer_source=normalizer_source)
			mosaic.focus(focus_list)
			mosaic.save(join(mosaic_root, mosaic_filename))
			mosaic.save_report(join(stats_root, sfutil.path_to_name(mosaic_filename)+'-mosaic_report.csv'))

		return AV, mosaic, umap

	def generate_mosaic_from_annotations(self, header_x, header_y, tile_px, tile_um, model=None, 
											mosaic_filename=None, umap_filename=None, outcome_label_headers=None, 
											filters=None, focus_filters=None, resolution='low', num_tiles_x=50,
											max_tiles_per_slide=100, expanded=False, use_optimal_tile=False, 
											activations_cache='default', normalizer=None, normalizer_source=None, 
											model_format=None):
		'''Generates a mosaic map by overlaying images onto a set of mapped tiles. 
			Slides are mapped using slide-level annotations, with x-axis value determined from header_x, and y-axis from header_y. 
			If use_optimal_tile is False and no model is provided, the first image tile in a slide's TFRecord is used for display.
			If optimal_tile is True, post-convolutional layer activations for all tiles in each slide are calculated using the provided model,
			and the tile nearest to centroid is used for display.
		
		Args:
			header_x:				Column name in annotations file from which to read X-axis coordinates.
			header_y:				Column name in annotations file from which to read Y-axis coordinates.
			tile_px:				Tile size in pixels.
			tile_um:				Tile size in microns.
			model:					Path to Tensorflow model to use when generating layer activations.
			mosaic_filename:		Filename for mosaic image. If not provided, mosaic will not be calculated or saved. Will be saved in project mosaic directory.
			umap_filename:			Filename for UMAP plot image. If not provided, plot will not be saved. Will be saved in project stats directory.
			outcome_label_headers:	Column name in annotations file from which to read category labels.
			filters:				Dataset filters to use when selecting slides to include the mosaic.
			focus_filters:			Dataset filters to use when selecting slides to highlight on the mosaic.
			resolution:				Resolution of the mosaic map. Impacts size of the final figure. Either low, medium, or high.
			num_tiles_x:			Specifies the size of the mosaic map grid.
			max_tiles_per_slide:	Limits the number of tiles taken from each slide. Too high of a number may introduce memory issues.
			expanded:				Bool. If False, will limit tile assignment to the corresponding grid space (strict display).
										If True, allows for display of nearby tiles if a given grid is empty.
			use_optimal_tile:		Bool. If True, will use model to create post-convolutional layer activations for all tiles in each slide,
										and choosing tile nearest to centroid for each slide for display.
			activations_cache:		Either 'default' or path to PKL file in which to save/cache nodal activations
			normalizer:				Normalization strategy to use on image tiles
			normalizer_source:		Path to normalizer source image
			model_format:			Optional. May supply format of saved Slideflow Keras model if the model was made with a legacy version.
										Default value will be slideflow.model.MODEL_FORMAT_CURRENT,
										but slideflow.model.MODEL_FORMAT_LEGACY may be supplied.
		'''
		from slideflow.activations import ActivationsVisualizer
		from slideflow.mosaic import Mosaic

		# Setup paths
		stats_root = join(self.PROJECT['root'], 'stats')
		mosaic_root = join(self.PROJECT['root'], 'mosaic')
		if not exists(stats_root): os.makedirs(stats_root)
		if not exists(mosaic_root): os.makedirs(mosaic_root)

		# Setup dataset
		dataset = self.get_dataset(filters=filters,
								   filter_blank=[header_x, header_y],
								   tile_px=tile_px,
								   tile_um=tile_um)

		# We are assembling a list of slides from the TFRecords path list, because we only want to use slides that have a corresponding TFRecord
		#  (some slides did not have a large enough ROI for tile extraction, and some slides may be in the annotations but are missing a slide image)
		slides = [sfutil.path_to_name(tfr) for tfr in dataset.get_tfrecords()]
		slide_labels_dict, _ = dataset.get_labels_from_annotations([header_x, header_y], use_float=True)
		slide_to_category = dataset.slide_to_label(outcome_label_headers)

		umap_x = np.array([slide_labels_dict[slide]['label'][0] for slide in slides])
		umap_y = np.array([slide_labels_dict[slide]['label'][1] for slide in slides])

		if use_optimal_tile and not model:
			log.error("Unable to calculate optimal tile if no model is specified.")
			return
		elif use_optimal_tile:
			# Calculate most representative tile in each slide/TFRecord for display
			AV = ActivationsVisualizer(model=model,
									   tfrecords=dataset.get_tfrecords(), 
									   root_dir=self.PROJECT['root'],
									   image_size=tile_px,
									   use_fp16=self.PROJECT['use_fp16'],
									   normalizer=normalizer,
									   normalizer_source=normalizer_source,
									   batch_size=self.FLAGS['eval_batch_size'],
									   max_tiles_per_slide=max_tiles_per_slide,
									   activations_cache='default',
									   model_format=model_format)

			optimal_slide_indices, _ = calculate_centroid(AV.slide_node_dict)

			# Restrict mosaic to only slides that had enough tiles to calculate an optimal index from centroid
			successful_slides = list(optimal_slide_indices.keys())
			num_warned = 0
			warn_threshold = 3
			for slide in slides:
				print_func = print if num_warned < warn_threshold else None
				if slide not in successful_slides:
					log.warn(f"Unable to calculate optimal tile for slide {sfutil.green(slide)}; will not include in Mosaic", 1, print_func)
					num_warned += 1
			if num_warned >= warn_threshold:
				log.warn(f"...{num_warned} total warnings, see {sfutil.green(log.logfile)} for details", 1)

			umap_x = np.array([slide_labels_dict[slide]['label'][0] for slide in successful_slides])
			umap_y = np.array([slide_labels_dict[slide]['label'][1] for slide in successful_slides])
			umap_meta = [{'slide': slide, 'index': optimal_slide_indices[slide]} for slide in successful_slides]
		else:
			# Take the first tile from each slide/TFRecord
			umap_meta = [{'slide': slide, 'index': 0} for slide in slides]

		umap = TFRecordMap.from_precalculated(tfrecords=dataset.get_tfrecords(),
											   slides=slides,
											   x=umap_x,
											   y=umap_y,
											   meta=umap_meta)

		mosaic_map = Mosaic(umap, 
							leniency=1.5,
							expanded=expanded,
							tile_zoom=15,
							num_tiles_x=num_tiles_x,
							tile_select='centroid' if use_optimal_tile else 'nearest',
							resolution=resolution,
							normalizer=normalizer,
							normalizer_source=normalizer_source)
		if mosaic_filename:
			mosaic_map.save(join(mosaic_root, mosaic_filename))
			mosaic_map.save_report(join(stats_root, sfutil.path_to_name(mosaic_filename)+'-mosaic_report.csv'))
		if umap_filename:
			umap.label_by_slide(slide_to_category)
			umap.save_2d_plot(join(stats_root, umap_filename))

	def generate_thumbnails(self, size=512, filters=None, filter_blank=None, roi=False, enable_downsample=False):
		'''Generates square slide thumbnails with black box borders of a fixed size, and saves to project folder.
		
		Args:
			size:				Int. Width/height of thumbnail in pixels.
			filters:			Dataset filters.
			filter_blank:		Header columns in annotations by which to filter slides, if the slides are blank in this column.
			enable_downsample:	Bool. If True and a thumbnail is not embedded in the slide file, downsampling is permitted in order
									to accelerate thumbnail calculation.
		'''
		import slideflow.slide as sfslide
		log.header('Generating thumbnails...')

		thumb_folder = join(self.PROJECT['root'], 'thumbs')
		if not exists(thumb_folder): os.makedirs(thumb_folder)
		dataset = self.get_dataset(filters=filters, filter_blank=filter_blank, tile_px=0, tile_um=0)
		slide_list = dataset.get_slide_paths()
		roi_list = dataset.get_rois()
		log.info(f"Saving thumbnails to {sfutil.green(thumb_folder)}", 1)

		for slide_path in slide_list:
			print(f"\r\033[KWorking on {sfutil.green(sfutil.path_to_name(slide_path))}...", end="")
			whole_slide = sfslide.SlideReader(slide_path, 
											  size_px=1000,
											  size_um=1000,
											  stride_div=1,
											  enable_downsample=enable_downsample,
											  roi_list=roi_list,
											  skip_missing_roi=roi,
											  buffer=None,
											  silent=True)
			if roi:
				thumb = whole_slide.annotated_thumb()
			else:
				thumb = whole_slide.square_thumb(size)
			thumb.save(join(thumb_folder, f'{whole_slide.name}.png'))
		print("\r\033[KThumbnail generation complete.")

	def generate_tfrecords_from_tiles(self, tile_px, tile_um, delete_tiles=True):
		'''Create tfrecord files from a collection of raw images, as stored in project tiles directory'''
		log.header('Writing TFRecord files...')

		# Load dataset for evaluation
		working_dataset = Dataset(config_file=self.PROJECT['dataset_config'],
								  sources=self.PROJECT['datasets'],
								  tile_px=tile_px,
								  tile_um=tile_um)
		
		for d in working_dataset.datasets:
			log.empty(f"Working on dataset {d}", 1)
			config = working_dataset.datasets[d]
			tfrecord_dir = join(config["tfrecords"], config['label'])
			tiles_dir = join(config["tiles"], config['label'])
			if not exists(tiles_dir):
				log.warn(f"No tiles found for dataset {sfutil.bold(d)}", 1)
				continue

			# Check to see if subdirectories in the target folders are slide directories (contain images)
			#  or are further subdirectories (e.g. validation and training)
			log.info('Scanning tile directory structure...', 2)
			if sfutil.contains_nested_subdirs(tiles_dir):
				subdirs = [_dir for _dir in os.listdir(tiles_dir) if isdir(join(tiles_dir, _dir))]
				for subdir in subdirs:
					tfrecord_subdir = join(tfrecord_dir, subdir)
					sfio.tfrecords.write_tfrecords_multi(join(tiles_dir, subdir), tfrecord_subdir)
			else:
				sfio.tfrecords.write_tfrecords_multi(tiles_dir, tfrecord_dir)

			working_dataset.update_manifest()

			if delete_tiles:
				shutil.rmtree(tiles_dir)
	
	def get_dataset(self, tile_px=None, tile_um=None, filters=None, filter_blank=None, verification='both'):
		'''Returns slideflow.io.Dataset object using project settings.

		Args:
			tile_px:		Tile size in pixels
			tile_um:		Tile size in microns
			filters:		Dictionary of annotations filters to use when selecting slides/TFRecords to include in dataset
			filter_blank:	List of label headers; will only include slides that are not blank in these headers
			verification:	'tfrecords', 'slides', or 'both'. If 'slides', will verify all annotations are mapped to slides.
															  If 'tfrecords', will check that TFRecords exist and update manifest
		'''
		try:
			dataset = Dataset(config_file=self.PROJECT['dataset_config'], 
							  sources=self.PROJECT['datasets'],
							  tile_px=tile_px,
							  tile_um=tile_um,
							  annotations=self.PROJECT['annotations'],
							  filters=filters,
							  filter_blank=filter_blank)

		except FileNotFoundError:
			log.warn("No datasets configured.")

		if verification in ('both', 'slides'):
			log.header("Verifying slide annotations...")
			dataset.verify_annotations_slides()
		if verification in ('both', 'tfrecords'):
			log.header("Verifying tfrecords and updating manifest...")
			dataset.update_manifest()

		return dataset

	def load_datasets(self, path):
		'''Loads datasets dictionaries from a given datasets.json file.'''
		try:
			datasets_data = sfutil.load_json(path)
			datasets_names = list(datasets_data.keys())
			datasets_names.sort()
		except FileNotFoundError:
			datasets_data = {}
			datasets_names = []
		return datasets_data, datasets_names

	def load_project(self, directory):
		'''Loads a saved and pre-configured project from the specified directory.'''
		if exists(join(directory, "settings.json")):
			self.PROJECT = sfutil.load_json(join(directory, "settings.json"))
			log.empty("Project configuration loaded.")
		else:
			raise OSError(f'Unable to locate settings.json at location "{directory}".')

		# Enable logging
		log.logfile = join(self.PROJECT['root'], "log.log")

		# Auto-update slidenames for newly added slides
		self.associate_slide_names()

	def resize_tfrecords(self, source_tile_px, source_tile_um, dest_tile_px, filters=None):
		'''Resizes images in a set of TFRecords to a given pixel size.

		Args:
			source_tile_px:		Pixel size of source images. Used to select source TFRecords.
			source_tile_um:		Micron size of source images. Used to select source TFRecords.
			dest_tile_px:		Pixel size of resized images.
			filters:			Dictionary of dataset filters to use for selecting TFRecords for resizing.
		'''
		log.header(f"Resizing TFRecord tiles to ({dest_tile_px}, {dest_tile_px})")
		resize_dataset = self.get_dataset(filters=filters,
										  tile_px=source_tile_px,
										  tile_um=source_tile_um)
		tfrecords_list = resize_dataset.get_tfrecords()
		log.info(f"Resizing {len(tfrecords_list)} tfrecords", 1)

		for tfr in tfrecords_list:
			sfio.tfrecords.transform_tfrecord(tfr, tfr+".transformed", resize=dest_tile_px)
	
	def extract_tiles_from_tfrecords(self, tile_px, tile_um, destination=None, filters=None):
		'''Extracts all tiles from a set of TFRecords.
		
		Args:
			tile_px:		Tile size in pixels
			tile_um:		Tile size in microns
			destination:	Destination folder in which to save tile images
			filters:		Dataset filters to use when selecting TFRecords
		'''
		log.header(f"Extracting tiles from TFRecords")
		to_extract_dataset = self.get_dataset(filters=filters,
											  tile_px=tile_px,
											  tile_um=tile_um)
		
		for dataset_name in self.PROJECT['datasets']:
			to_extract_tfrecords = to_extract_dataset.get_tfrecords(dataset=dataset_name)
			if destination:
				tiles_dir = destination
			else:
				tiles_dir = join(to_extract_dataset.datasets[dataset_name]['tiles'], to_extract_dataset.datasets[dataset_name]['label'])
				if not exists(tiles_dir):
					os.makedirs(tiles_dir)
			for tfr in to_extract_tfrecords:
				sfio.tfrecords.extract_tiles(tfr, tiles_dir)		

	def save_project(self):
		'''Saves current project configuration as "settings.json".'''
		sfutil.write_json(self.PROJECT, join(self.PROJECT['root'], 'settings.json'))

	def train(self, model_names=None, outcome_label_headers='category', input_header=None, filters=None, filter_blank=None,
				resume_training=None, checkpoint=None, pretrain='imagenet', pretrain_model_format=None, batch_file=None,
				hyperparameters=None, validation_target=None, validation_strategy=None,validation_fraction=None,
				validation_k_fold=None, k_fold_iter=None, k_fold_header=None, validation_dataset=None, validation_annotations=None,
				validation_filters=None, validate_on_batch=512, validation_steps=200, max_tiles_per_slide=0, min_tiles_per_slide=0,
				starting_epoch=0, steps_per_epoch_override=None, auto_extract=False, normalizer=None, 
				normalizer_source=None, normalizer_strategy='tfrecord', use_tensorboard=False, multi_gpu=False, save_predictions=False,
				skip_metrics=False):

		'''Train model(s).

		Args:
			model_names:			Either a string representing a model name, or an array of strings containing multiple model names. 
										Required if training to a single hyperparameter combination with the "hyperparameters" argument.
										If performing a hyperparameter sweep, will only train models with these names in the batch_train.tsv config file.
										May supply None if performing a hyperparameter sweep, in which case all models in the batch_train.tsv config file will be trained.
			outcome_label_headers:	String or list. Specifies which header(s) in the annotation file to use for the output category. 
										Defaults to 'category'.	If a list is provided, will loop through all outcomes and perform HP sweep on each.
			filters:				Dictionary of column names mapping to column values by which to filter slides using the annotation file.
			resume_training:		Path to Tensorflow model to continue training
			checkpoint:				Path to cp.ckpt from which to load weights
			pretrain:				Pretrained weights to load. Default is imagenet. May supply a compatible Tensorflow model from which to load weights.
			pretrain_model_format:	Optional. May supply format of pretrained Slideflow Keras model if the model was made with a legacy version.
										Default value will be slideflow.model.MODEL_FORMAT_CURRENT,
										but slideflow.model.MODEL_FORMAT_LEGACY may be supplied.
			batch_file:				Manually specify batch file to use for a hyperparameter sweep. If not specified, will use project default.
			hyperparameters:		Manually specify hyperparameter combination to use for training. If specified, will ignore batch training file.
			validation_target: 		Whether to select validation data on a 'per-patient' or 'per-tile' basis. If not specified, will use project default.
			validation_strategy:	Validation dataset selection strategy (bootstrap, k-fold, k-fold-manual, k-fold-preserved-site, fixed, none). If not specified, will use project default.
			validation_fraction:	Fraction of data to use for validation testing. If not specified, will use project default.
			validation_k_fold: 		K, if using k-fold validation. If not specified, will use project default.
			k_fold_iter:			Which iteration to train if using k-fold validation. Defaults to training all iterations.
			k_fold_header:			Annotations file header column for manually specifying k-fold. Only used if validation_strategy is 'k-fold-manual'
			validation_dataset:		If specified, will use a separate dataset on which to perform validation.
			validation_annotations:	If using a separate dataset for validation, the annotations CSV must be supplied.
			validation_filters:		If using a separate dataset for validation, these filters are used to select a subset of slides for validation.
			validate_on_batch:		Validation will be performed every X batches.
			validation_steps:		Number of batches to use for each instance of validation
			max_tiles_per_slide:	Will only use up to this many tiles from each slide for training. If zero, will include all tiles.
			min_tiles_per_slide:	Minimum number of tiles a slide must have to be included in training. 
			starting_epoch:			Starts training at the specified epoch
			steps_per_epoch_override:	If provided, will manually set the number of steps in an epoch (default epoch length is the number of total tiles)
			auto_extract:			Bool. If True, will automatically extract tiles as needed for training, without needing to explicitly call extract_tiles()
			normalizer:				Normalization strategy to use on image tiles
			normalizer_source:		Path to normalizer source image
			normalizer_strategy:	Either 'tfrecord' or 'realtime'. If TFrecord and auto_extract is True, then tiles will be extracted to TFRecords and stored normalized.
										If realtime, then normalization is performed during training.
			use_tensorboard:		Bool. If True, will add tensorboard callback during training for realtime monitoring.
			multi_gpu:				Bool. If True, will attempt to train using multiple GPUs using Keras MirroredStrategy.
			
		Returns:
			A dictionary containing model names mapped to train_acc, val_loss, and val_acc
		'''

		assert not (k_fold_header is None and validation_strategy == 'k-fold-manual'), "Must supply 'k_fold_header' if validation strategy is 'k-fold-manual'"

		# Reconcile provided arguments with project defaults
		batch_train_file = self.PROJECT['batch_train_config'] if not batch_file else join(self.PROJECT['root'], batch_file)
		validation_strategy = self.PROJECT['validation_strategy'] if not validation_strategy else validation_strategy
		validation_target = self.PROJECT['validation_target'] if not validation_target else validation_target
		validation_fraction = self.PROJECT['validation_fraction'] if not validation_fraction else validation_fraction
		validation_k_fold = self.PROJECT['validation_k_fold'] if not validation_k_fold else validation_k_fold
		validation_log = join(self.PROJECT['root'], "validation_plans.json")

		# Quickly scan for errors (duplicate model names in batch training file) and prepare models to train
		if hyperparameters and not model_names:
			log.error("If specifying hyperparameters, 'model_names' must be supplied. ", 1)
			return
		if normalizer and normalizer_strategy not in ('tfrecord', 'realtime'):
			log.error(f"Unknown normalizer strategy {normalizer_strategy}, must be either 'tfrecord' or 'realtime'", 1)
			return
		if validation_strategy in ('k-fold-manual', 'k-fold-preserved-site', 'k-fold', 'bootstrap') and validation_dataset:
			log.error(f"Unable to use {validation_strategy} if validation_dataset has been provided.", 1)
			return

		# Setup normalization
		tfrecord_normalizer = normalizer if (normalizer and normalizer_strategy == 'tfrecord') else None
		tfrecord_normalizer_source = normalizer_source if (normalizer and normalizer_strategy == 'tfrecord') else None
		train_normalizer = normalizer if (normalizer and normalizer_strategy == 'realtime') else None
		train_normalizer_source = normalizer_source if (normalizer and normalizer_strategy == 'realtime') else None

		# Prepare hyperparameters
		log.header("Performing hyperparameter sweep...")
		
		hyperparameter_list = self._get_hyperparameter_combinations(hyperparameters, model_names, batch_train_file)

		outcome_label_headers = [outcome_label_headers] if not isinstance(outcome_label_headers, list) else outcome_label_headers
		if len(outcome_label_headers) > 1:
			log.info(f"Training ({len(hyperparameter_list)} models) using {len(outcome_label_headers)} variables as simultaneous outcomes:", 1)
			for label in outcome_label_headers:
				log.empty(label, 2)
			if log.INFO_LEVEL > 0: print()

		# Next, prepare the multiprocessing manager (needed to free VRAM after training and keep track of results)
		manager = multiprocessing.Manager()
		results_dict = manager.dict()
		ctx = multiprocessing.get_context('spawn')

		# For each hyperparameter combination, perform training
		for hp, hp_model_name in hyperparameter_list:

			# Prepare k-fold validation configuration
			results_log_path = os.path.join(self.PROJECT['root'], "results_log.csv")
			k_fold_iter = [k_fold_iter] if (k_fold_iter != None and not isinstance(k_fold_iter, list)) else k_fold_iter

			if validation_strategy == 'k-fold-manual':
				training_dataset = self.get_dataset(tile_px=hp.tile_px, 
													tile_um=hp.tile_um,
													filters=filters,
													filter_blank=filter_blank)

				k_fold_slide_labels, valid_k = training_dataset.slide_to_label(k_fold_header, return_unique=True)
				k_fold = len(valid_k)
			else:
				k_fold = validation_k_fold if validation_strategy in ('k-fold', 'k-fold-preserved-site', 'bootstrap') else 0
				valid_k = [] if not k_fold else [kf for kf in range(1, k_fold+1) if ((k_fold_iter and kf in k_fold_iter) or (not k_fold_iter))]
				k_fold_slide_labels = None

			if hp.model_type() != 'linear' and len(outcome_label_headers) > 1:
				#raise Exception("Multiple outcome labels only supported for linear outcome labels.")
				log.info("Using experimental multi-outcome approach for categorical outcome")

			# Auto-extract tiles if requested
			if auto_extract:
				self.extract_tiles(hp.tile_px,
									hp.tile_um,
									filters=filters,
									filter_blank=filter_blank,
									normalizer=tfrecord_normalizer,
									normalizer_source=tfrecord_normalizer_source)

			label_string = "-".join(outcome_label_headers)
			model_name = f"{label_string}-{hp_model_name}"
			model_iterations = [model_name] if not k_fold else [f"{model_name}-kfold{k}" for k in valid_k]

			def start_training_process(k):
				# Using a separate process ensures memory is freed once training has completed
				process = ctx.Process(target=project_utils.trainer, args=(outcome_label_headers, model_name, self.PROJECT,
															results_dict, hp, validation_strategy,  validation_target,
															validation_fraction, validation_k_fold,  validation_log, validation_dataset, validation_annotations,
															validation_filters, k, k_fold_slide_labels, input_header, filters, filter_blank, pretrain,
															pretrain_model_format, resume_training, checkpoint, validate_on_batch, validation_steps,
															max_tiles_per_slide, min_tiles_per_slide, starting_epoch, steps_per_epoch_override, train_normalizer, 
															train_normalizer_source, use_tensorboard, multi_gpu, save_predictions, skip_metrics, self.FLAGS))
				process.start()
				log.empty(f"Spawning training process (PID: {process.pid})")
				process.join()

			# Perform training
			log.header("Training model...")
			if k_fold:
				for k in valid_k:
					start_training_process(k)
					
			else:
				start_training_process(None)

			# Record results
			for mi in model_iterations:
				if mi not in results_dict:
					log.error(f"Training failed for model {model_name}")
				else:
					sfutil.update_results_log(results_log_path, mi, results_dict[mi]['epochs'])
			log.complete(f"Training complete for model {model_name}, results saved to {sfutil.green(results_log_path)}")

		# Print summary of all models
		log.complete("Training complete; validation accuracies:", 0)
		for model in results_dict:
			try:
				last_epoch = max([int(e.split('epoch')[-1]) for e in results_dict[model]['epochs'].keys() if 'epoch' in e ])
				final_train_metrics = results_dict[model]['epochs'][f'epoch{last_epoch}']['train_metrics']
				final_val_metrics = results_dict[model]['epochs'][f'epoch{last_epoch}']['val_metrics']
				log.empty(f"{sfutil.green(model)} training metrics:", 1)
				for m in final_train_metrics:
					log.empty(f"{m}: {final_train_metrics[m]}", 2)
				log.empty(f"{sfutil.green(model)} validation metrics:", 1)
				for m in final_val_metrics:
					log.empty(f"{m}: {final_val_metrics[m]}", 2)
			except ValueError:
				pass

		return results_dict

	def train_clam(self, exp_name, outcome_label_headers, model=None, pt_files='auto', num_features=None, filters=None, filter_blank=None, 
					activation_layers=['postconv'],	max_tiles_per_slide=0, min_tiles_per_slide=16, train_slides='same', validation_slides='same',
					tile_px=None, tile_um=None, k=1, k_start=-1, k_end=-1, max_epochs=20, lr=1e-4, label_frac=1, reg=1e-5, early_stopping=False, opt='adam',
					drop_out=False, bag_loss='ce', bag_weight=0.7, model_type='clam_sb', weighted_sample=False, no_inst_cluster=False, inst_loss=None,
					subtyping=False, B=8, attention_heatmaps=True, force_regenerate_features=False):

		'''Using a trained model, generate feature activations and train a CLAM model.
		
		Args:
			exp_name
			model
			outcome_label_headers
			pt_files
			filters
			filter_blank
			max_tiles_per_slide
			min_tiles_per_slide
			train_src
			val_src
			k
			k_start
			k_end
			max_epochs
			lr
			label_frac
			reg
			early_stopping
			opt
			drop_out
			bag_loss
			bag_weight
			model_type
			weighted_sample
			no_inst_cluster
			inst_loss
			subtyping
			B
			
		Returns:
			None
		
		New requirements:
			torch
			torchvision
		'''

		import slideflow.clam as clam
		from slideflow.clam.datasets.dataset_generic import Generic_MIL_Dataset
		from slideflow.clam.create_attention import export_attention
		from slideflow.io.tfrecords import get_tfrecords_from_model_manifest

		assert min_tiles_per_slide > 8, "Slides must have at least 8 tiles to train CLAM."
		assert model is not None or pt_files != 'auto', 'Must supply either a valid model to generate activations, or a path to pt_files'
		assert not (model is None and num_features is None), 'If supplying pre-generated activations via pt_files, must specify "num_features"'
		assert not (model is None and (train_slides == 'same' or validation_slides == 'same')), 'Must supply valid slide list with "train_slides" and "validation_slides" if not supplying a model.'
		assert not (model is None and (tile_px is None or tile_um is None)), 'If supplying pre-generated activations via pt_files, must specify "tile_px" and "tile_um"'

		# Set up CLAM experiment data directory
		clam_dir = join(self.PROJECT['root'], 'clam', exp_name)
		results_dir = join(clam_dir, 'results')
		if not exists(results_dir): os.makedirs(results_dir)

		if model is not None:
			# First, ensure the model is valid with a hyperparameters file
			try:
				hp_data = sfutil.load_json(join(dirname(model), 'hyperparameters.json'))
				tile_px = hp_data['tile_px']
				tile_um = hp_data['tile_um']
			except FileNotFoundError:
				raise Exception('Unable to find model hyperparameters file.')

			# Set up the pt_files directory for storing model activations
			if pt_files.lower() == 'auto':
				model_name_end = '' if 'k_fold_i' not in hp_data else f'_kfold{hp_data["k_fold_i"]}'
				pt_files = join(self.PROJECT['root'], 'pt_files', hp_data['model_name']+model_name_end)
			if not exists(pt_files):
				os.makedirs(pt_files)

			# Detect already generated pt files
			already_generated = [sfutil.path_to_name(f) for f in os.listdir(pt_files) if sfutil.path_to_ext(join(pt_files, f)) == 'pt']
			if force_regenerate_features or not len(already_generated):
				activation_filters = filters
			else:
				pt_dataset = self.get_dataset(tile_px, tile_um, filters=filters, filter_blank=filter_blank)
				all_slides = pt_dataset.get_slides()
				slides_to_generate = [s for s in all_slides if s not in already_generated]
				activation_filters = filters.copy()
				activation_filters['slide'] = slides_to_generate
				filtered_dataset = self.get_dataset(tile_px, tile_um, filters=activation_filters, filter_blank=filter_blank)
				filtered_slides_to_generate = filtered_dataset.get_slides()
				log.info(f'Activations already generated for {len(already_generated)} files, will not regenerate for these.', 1)
				log.info(f'Attempting to generate for {len(filtered_slides_to_generate)} slides', 1)

			# Set up activations interface
			activations_results = self.generate_activations(model,
															filters=activation_filters,
															filter_blank=filter_blank,
															layers=activation_layers,
															max_tiles_per_slide=max_tiles_per_slide,
															min_tiles_per_slide=min_tiles_per_slide,
															torch_export=pt_files,
															isolated_thread=True,
															activations_cache=None)

			# Export activations to pt_files folder in torch format
			num_features = activations_results['num_features']

		model_size = [num_features,256,128]

		# Set up training/validation splits (mirror base model)
		split_dir = join(clam_dir, 'splits')
		if not exists(split_dir): os.makedirs(split_dir)

		# Auto-detect training/validation slides from model if desired
		if train_slides == 'same':
			try:
				train_slides = get_tfrecords_from_model_manifest(join(dirname(model), 'slide_manifest.log'), dataset='training')
				train_slides = [s for s in train_slides if exists(join(pt_files, s+'.pt'))]
			except FileNotFoundError:
				raise Exception("Unable to auto-detect training/validation split from source model, 'slide_manifest.log' not found in model directory.")

		if validation_slides == 'same':
			try:
				validation_slides = get_tfrecords_from_model_manifest(join(dirname(model), 'slide_manifest.log'), dataset='validation')
				validation_slides = [s for s in validation_slides if exists(join(pt_files, s+'.pt'))]
			except FileNotFoundError:
				raise Exception("Unable to auto-detect training/validation split from source model, 'slide_manifest.log' not found in model directory.")

		header = ['','train','val','test']
		with open(join(split_dir, 'splits_0.csv'), 'w') as splits_file:
			writer = csv.writer(splits_file)
			writer.writerow(header)
			for i in range(max(len(train_slides), len(validation_slides))):
				row = [i]
				if i < len(train_slides): 		row += [train_slides[i]]
				else: 							row += ['']
				if i < len(validation_slides):	row += [validation_slides[i], validation_slides[i]]	# Currently, this sets the validation & test sets in CLAM to be the same
				else:							row += ['', '']
				writer.writerow(row)

		# Set up outcomes for CLAM model
		dataset = self.get_dataset(tile_px=tile_px,
								   tile_um=tile_um,
								   filters=filters,
								   filter_blank=filter_blank)

		slide_labels, unique_labels = dataset.get_labels_from_annotations(outcome_label_headers, 
																		  use_float=False,		 # CLAM only supports categorical outcomes
																		  key='outcome_label')		
		# Set up CLAM args/settings
		args_dict = {
			'num_splits': k,
			'k': k,
			'k_start': k_start,
			'k_end': k_end,
			'max_epochs': max_epochs,
			'lr': lr,
			'reg': reg,
			'label_frac': label_frac,
			'bag_loss': bag_loss,
			'bag_weight': bag_weight,
			'model_type': model_type,
			'model_size': model_size,
			'use_drop_out': drop_out,
			'drop_out': drop_out,
			'weighted_sample': weighted_sample,
			'opt': opt,
			'inst_loss': inst_loss,
			'no_inst_cluster': no_inst_cluster,
			'B': B,								 
			'split_dir': split_dir,
			'data_root_dir': pt_files,
			'log_data': False,
			'testing': False,
			'early_stopping': early_stopping,
			'subtyping': subtyping,
			'seed': 1,
			'results_dir': results_dir,
			'n_classes': len(unique_labels)
		}
		args = types.SimpleNamespace(**args_dict)
		sfutil.write_json(args_dict, join(clam_dir, 'experiment.json'))

		# Create CLAM dataset
		clam_dataset = Generic_MIL_Dataset(csv_path=self.PROJECT['annotations'],
										   data_dir=pt_files,
										   shuffle=False,
										   seed=args.seed,
										   print_info=True,
										   label_col = outcome_label_headers,
										   label_dict = dict(zip(unique_labels, range(len(unique_labels)))),
										   patient_strat=False,
										   ignore=[])

		# Run CLAM
		clam.main(args, clam_dataset)

		# Get attention from trained model on validation set
		attention_tfrecords = [tfr for tfr in dataset.get_tfrecords() if sfutil.path_to_name(tfr) in validation_slides]
		for ki in range(k):
			attention_dir = join(clam_dir, 'attention', str(ki))
			if not exists(attention_dir): os.makedirs(attention_dir)
			export_attention(args_dict, 
							 ckpt_path=join(results_dir, f's_{ki}_checkpoint.pt'),
							 export_dir=attention_dir,
							 pt_files=pt_files,
							 slides=validation_slides,
							 reverse_label_dict = dict(zip(range(len(unique_labels)), unique_labels)),
							 slide_to_label = {s:slide_labels[s]['outcome_label'] for s in slide_labels})
			if attention_heatmaps:
				heatmaps_dir = join(clam_dir, 'attention_heatmaps', str(ki))
				if not exists(heatmaps_dir): os.makedirs(heatmaps_dir)
				
				for tfr in attention_tfrecords:
					attention_dict = {}
					slide = sfutil.path_to_name(tfr)
					try:
						with open(join(attention_dir, slide+'.csv'), 'r') as csv_file:
							reader = csv.reader(csv_file)
							for row in reader:
								attention_dict.update({int(row[0]): float(row[1])})
					except FileNotFoundError:
						print(f"Unable to find attention scores for slide {slide}, skipping")
						continue
					self.generate_tfrecord_heatmap(tfr, attention_dict, heatmaps_dir, tile_px=tile_px, tile_um=tile_um)
		
	def evaluate_clam(self, trained_exp, outcome_label_headers, eval_tag=None, pt_files='auto', num_features=None, filters=None, filter_blank=None,
						activation_layers=['postconv'], max_tiles_per_slide=0, min_tiles_per_slide=16, attention_heatmaps=True, tile_px=None, tile_um=None):
		
		import slideflow.clam as clam
		from slideflow.clam.datasets.dataset_generic import Generic_MIL_Dataset
		from slideflow.clam.create_attention import export_attention

		# Detect source CLAM experiment which we are evaluating.
		# First, assume it lives in this project's clam folder
		if exists(join(self.PROJECT['root'], 'clam', trained_exp, 'experiment.json')):
			trained_exp = join(self.PROJECT['root'], 'clam', trained_exp)
		elif exists(join(trained_exp, 'experiment.json')):
			pass
		else:
			raise Exception(f"Unable to find the experiment '{trained_exp}'")
		
		log.info(f"Loading trained experiment from {sfutil.green(trained_exp)}", 1)
		eval_dir = join(trained_exp, 'eval')
		if not exists(eval_dir): os.makedirs(eval_dir)

		# Set up evaluation directory with unique evaluation tag
		existing_tags = [int(d) for d in os.listdir(eval_dir) if d.isdigit()]
		if eval_tag is None:
			eval_tag = '0' if not existing_tags else str(max(existing_tags))

		# Ensure evaluation tag will not overwrite existing results
		if eval_tag in existing_tags:
			unique, base_tag = 1, eval_tag
			eval_tag = f'{base_tag}_{unique}'
			while exists(join(eval_dir, eval_tag)):
				eval_tag = f'{base_tag}_{unique}'
				unique += 1
			log.info(f"Eval tag {base_tag} already exists, will save evaluation under 'eval_tag'")

		# Load or generate activations:
		if pt_files == 'auto':
			...

		# Load trained model checkpoint
		ckpt_path = join(trained_exp, 'results', 's_0_checkpoint.pt')
		eval_dir = join(eval_dir, eval_tag)
		if not exists(eval_dir): os.makedirs(eval_dir)
		args_dict = sfutil.load_json(join(trained_exp, 'experiment.json'))
		args = types.SimpleNamespace(**args_dict)
		args.save_dir = eval_dir

		dataset = self.get_dataset(tile_px=tile_px,
								   tile_um=tile_um,
								   filters=filters,
								   filter_blank=filter_blank)

		evaluation_slides = [s for s in dataset.get_slides() if exists(join(pt_files, s+'.pt'))]
		dataset.apply_filters({'slide': evaluation_slides})

		slide_labels, unique_labels = dataset.get_labels_from_annotations(outcome_label_headers,
																		  use_float=False,
																		  key='outcome_label')
		
		# Set up evaluation annotations file based off existing pt_files
		outcome_dict = dict(zip(range(len(unique_labels)), unique_labels))
		with open(join(eval_dir, 'eval_annotations.csv'), 'w') as eval_file:
			writer = csv.writer(eval_file)
			header = ['submitter_id', 'slide', outcome_label_headers]
			writer.writerow(header)
			for slide in evaluation_slides:
				row = [slide, slide, outcome_dict[slide_labels[slide]['outcome_label']]]
				writer.writerow(row)

		clam_dataset = Generic_MIL_Dataset(csv_path=join(eval_dir, 'eval_annotations.csv'),
										   data_dir=pt_files,
										   shuffle=False,
										   seed=args.seed,
										   print_info=True,
										   label_col=outcome_label_headers,
										   label_dict = dict(zip(unique_labels, range(len(unique_labels)))),
										   patient_strat=False,
										   ignore=[])
		
		clam.evaluate(ckpt_path, args, clam_dataset)

		# Get attention from trained model on validation set
		attention_tfrecords = dataset.get_tfrecords()
		attention_dir = join(eval_dir, 'attention')
		if not exists(attention_dir): os.makedirs(attention_dir)
		export_attention(args_dict, 
							ckpt_path=ckpt_path,
							export_dir=attention_dir,
							pt_files=pt_files,
							slides=dataset.get_slides(),
							reverse_label_dict = dict(zip(range(len(unique_labels)), unique_labels)),
							slide_to_label = {s:slide_labels[s]['outcome_label'] for s in slide_labels})
		if attention_heatmaps:
			heatmaps_dir = join(eval_dir, 'attention_heatmaps')
			if not exists(heatmaps_dir): os.makedirs(heatmaps_dir)
			
			for tfr in attention_tfrecords:
				attention_dict = {}
				slide = sfutil.path_to_name(tfr)
				try:
					with open(join(attention_dir, slide+'.csv'), 'r') as csv_file:
						reader = csv.reader(csv_file)
						for row in reader:
							attention_dict.update({int(row[0]): float(row[1])})
				except FileNotFoundError:
					print(f"Unable to find attention scores for slide {slide}, skipping")
					continue
				self.generate_tfrecord_heatmap(tfr, attention_dict, heatmaps_dir, tile_px=tile_px, tile_um=tile_um)

	def generate_tfrecord_heatmap(self, tfrecord, tile_dict, export_dir, tile_px, tile_um):
		'''Creates a tfrecord-based WSI heatmap using a dictionary of tile values for heatmap display. '''
		
		from slideflow.io.tfrecords import get_locations_from_tfrecord
		from slideflow.slide import SlideReader

		slide_name = sfutil.path_to_name(tfrecord)
		loc_dict = get_locations_from_tfrecord(tfrecord)
		dataset = self.get_dataset(tile_px=tile_px, tile_um=tile_um)
		slide_paths = {sfutil.path_to_name(sp):sp for sp in dataset.get_slide_paths()}
		
		try:
			slide_path = slide_paths[slide_name]
		except KeyError:
			raise Exception(f"Unable to locate slide {slide_name}")

		if tile_dict.keys() != loc_dict.keys():
			raise Exception(f"Length of provided tile_dict ({len(list(tile_dict.keys()))}) does not match number of tiles stored in the TFRecord ({len(list(loc_dict.keys()))}).")

		print(f"Generating TFRecord heatmap for {sfutil.green(tfrecord)}...")
		slide = SlideReader(slide_path, tile_px, tile_um, skip_missing_roi=False)

		stats = {}

		# Loaded CSV coordinates:
		x = [int(loc_dict[l][0]) for l in loc_dict]
		y = [int(loc_dict[l][1]) for l in loc_dict]
		vals = [tile_dict[l] for l in loc_dict]

		stats.update({
			slide_name: {
				'mean':mean(vals),
				'median':median(vals),
				'above_0':len([v for v in vals if v > 0]),
				'above_1':len([v for v in vals if v > 1]),
			}
		})

		print("\nLoaded tile values")
		print(f"Min: {min(vals)}\t Max:{max(vals)}")

		scaled_x = [(xi * slide.ROI_SCALE) - slide.full_extract_px/2 for xi in x]
		scaled_y = [(yi * slide.ROI_SCALE) - slide.full_extract_px/2 for yi in y]

		print("\nLoaded CSV coordinates:")
		print(f"Min x: {min(x)}\t Max x: {max(x)}")
		print(f"Min y: {min(y)}\t Max y: {max(y)}")

		print("\nScaled CSV coordinates:")
		print(f"Min x: {min(scaled_x)}\t Max x: {max(scaled_x)}")
		print(f"Min y: {min(scaled_y)}\t Max y: {max(scaled_y)}")

		print("\nSlide properties:")
		print(f"Raw size (x): {slide.full_shape[0]}\t Raw size (y): {slide.full_shape[1]}")

		# Slide coordinate information
		max_coord_x = max([c[0] for c in slide.coord])
		max_coord_y = max([c[1] for c in slide.coord])
		num_x = len(set([c[0] for c in slide.coord]))
		num_y = len(set([c[1] for c in slide.coord]))

		print("\nSlide tile grid:")
		print(f"Number of tiles (x): {num_x}\t Max coord (x): {max_coord_x}")
		print(f"Number of tiles (y): {num_y}\t Max coord (y): {max_coord_y}")

		# Calculate dead space (un-extracted tiles) in x and y axes
		dead_x = slide.full_shape[0] - max_coord_x
		dead_y = slide.full_shape[1] - max_coord_y
		fraction_dead_x = dead_x / slide.full_shape[0]
		fraction_dead_y = dead_y / slide.full_shape[1]

		print("\nSlide dead space")
		print(f"x: {dead_x}\t y:{dead_y}")

		# Work on grid
		x_grid_scale = max_coord_x / (num_x-1)
		y_grid_scale = max_coord_y / (num_y-1)

		print("\nCoordinate grid scale:")
		print(f"x: {x_grid_scale}\t y: {y_grid_scale}")

		grid = np.zeros((num_y, num_x))

		indexed_x = [round(xi / x_grid_scale) for xi in scaled_x]
		indexed_y = [round(yi / y_grid_scale) for yi in scaled_y]

		for i, (xi,yi,v) in enumerate(zip(indexed_x,indexed_y,vals)):
			grid[yi][xi] = v

		fig = plt.figure(figsize=(18, 16))
		ax = fig.add_subplot(111)
		fig.subplots_adjust(bottom = 0.25, top=0.95)
		gca = plt.gca()
		gca.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

		print("Generating thumbnail...")
		thumb = slide.thumb(mpp=5)
		print("Saving thumbnail....")
		thumb.save(join(export_dir, f'{slide_name}' + '.png'))
		print("Generating figure...")
		implot = ax.imshow(thumb, zorder=0)

		extent = implot.get_extent()
		extent_x = extent[1]
		extent_y = extent[2]
		grid_extent = (extent[0], extent_x * (1-fraction_dead_x), extent_y * (1-fraction_dead_y), extent[3])

		print("\nImage extent:")
		print(extent)
		print("\nGrid extent:")
		print(grid_extent)

		divnorm=mcol.TwoSlopeNorm(vmin=min(-0.01, min(vals)), vcenter=0, vmax=max(0.01, max(vals)))
		heatmap = ax.imshow(grid, zorder=10, alpha=0.6, extent=grid_extent, interpolation='bicubic', cmap='coolwarm', norm=divnorm)

		print("Saving figure...")
		plt.savefig(join(export_dir, f'{slide_name}_attn.png'), bbox_inches='tight')

		# Clean up
		print("Cleaning up...")
		plt.clf()
		del slide
		del thumb

		return stats

	def visualize_tiles(self, model, node, tfrecord_dict=None, directory=None, mask_width=None, 
						normalizer=None, normalizer_source=None, model_format=None):
		'''Visualizes node activations across a set of image tiles through progressive convolutional masking.

		Args:
			model:				Path to Tensorflow model
			node:				Int, node to analyze
			tfrecord_dict:		Dictionary mapping tfrecord paths to tile indices. Visualization will be performed on these tiles.
			directory:			Directory in which to save images.
			mask_width:			Width of mask to convolutionally apply. Defaults to 1/6 of tile_px
			normalizer:				Normalization strategy to use on image tiles.
			normalizer_source:		Path to normalizer source image.
			model_format:		Optional. May supply format of saved Slideflow Keras model if the model was made with a legacy version.
									Default value will be slideflow.model.MODEL_FORMAT_CURRENT,
									but slideflow.model.MODEL_FORMAT_LEGACY may be supplied.
		'''
		from slideflow.activations import TileVisualizer

		hp_data = sfutil.load_json(join(dirname(model), 'hyperparameters.json'))
		tile_px = hp_data['hp']['tile_px']
		TV = TileVisualizer(model=model, 
							node=node,
							tile_px=tile_px,
							mask_width=mask_width,
							normalizer=normalizer,
							normalizer_source=normalizer_source,
							model_format=model_format)

		if tfrecord_dict:
			for tfrecord in tfrecord_dict:
				for tile_index in tfrecord_dict[tfrecord]:
					TV.visualize_tile(tfrecord=tfrecord, index=tile_index, export_folder=directory)

		else:
			tiles = [o for o in os.listdir(directory) if not isdir(join(directory, o))]
			tiles.sort(key=lambda x: int(x.split('-')[0]))
			tiles.reverse()
			for tile in tiles[:20]:
				tile_loc = join(directory, tile)
				TV.visualize_tile(image_jpg=tile_loc, export_folder=directory)