# Image processing programs for SHG FROG data
# By Adam Fisher, est 7/11/22

# packages/boilerplate
import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True)
mpl.rc('ytick', direction='in', right=True)
mpl.rc('xtick.minor', visible=True)
mpl.rc('ytick.minor', visible=True)
%matplotlib inline

# constants
c = 2.99e8 # speed of light in vac [m/s]
# factors for plotting, b/c im working in m and sec
h2th = 1.e-15 # Hz to THz
m2nm = 1.e9 # m to nm
s2fs = 1.e15 # s to fs

# first program is to take in raw pic and calibration stuff
def read_trace(folder,fname,calibration):
	"""
	Reads in the raw FROG photo and the calibration data and outputs the raw trace and its associated arrays
	supports image formats that are supported by cv2.imread, consult documentation if unsure
	requires calibration data in a dict, order doesnt matter but name and units do, please follow convention
	calibration wavelength [m] (float), calibration pixel [pix] (int), delay per pixel [s/pix] (float), wavelength per pixel [m/pix] (float)
	note: horz pix is wvl and vert is delay
	note: may wish to import data to a dict in future for greater flexiblity
	inputs:
	folder - str - relative file path to directory that contains desired FROG photo
	fname - str - file name of FROG photo you wish to use
	calibration - dict - dict of calibration data that matches FROG photo, please see format above
	NOTE: photo and calibration file must be in same directory
	output:
	raw_trace - array, NxM - array containing the intensity data from FROG photo, I_FROG(omega_i,tau_j) from Trebino's book
	trace_delay - array, Mx1 - array containing the delays 
	trace_wvl - array, Nx1 - array containing the wavelengths recorded by the FROG
	trace_freq - array, Nx1 - SAA but the frequencies instead
	"""
	# assertions for files
	assert(type(folder)==str), 'input: folder, must be string'
	assert(type(fname)==str), 'input: fname, must be string'
	assert(type(calibration)==dict), 'input: calibration, must be a dict'
	# load in calibration
	cali_wvl = float(calibration['calibration wavelength']) # [m]
	cali_pix = int(calibration['calibration pixel']) # [pix]
	dpp = float(calibration['delay per pixel']) # [s/pix]
	wpp = float(calibration['wavelength per pixel']) # [m/pix]
	# load in picture
	fp = folder+fname
	try:
		with open(fp,'r',) as f:
			raw_pic = cv.imread(f,cv.IMREAD_UNCHANGED)
	except:
		raise Exception('error reading picture')
	# create trace arrays, will need to be able to handle even and odd image dim
	# create trace delay array
	if bool(raw_pic.shape[0]%2): # true = odd
		trc_d = np.linspace((1-raw_pic.shape[0]/2))