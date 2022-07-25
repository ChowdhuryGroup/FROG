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
	output: as tuple in this order
	raw_trace - array, NxM - array containing the intensity data from FROG photo, I_FROG(omega_i,tau_j) from Trebino's book
	NOTE: N and M are the number of delay points and wavelengths that the FROG setup measured, respectively
	trace_delay - array, Nx1 - array containing the delays [s]
	trace_wvl - array, Mx1 - array containing the wavelengths recorded by the FROG [m]
	trace_freq - array, Mx1 - SAA but the frequencies instead [Hz]
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
	# create trace arrays, dim dont matter until we have to crop and such
	# delay is vert, wvl is horz, pic dim is vert pix x horz pix
	# create trace delay array [s]
	trc_d = np.linspace((-raw_pic.shape[0]/2.),(raw_pic.shape[0]/2.),raw_pic.shape[0])*dpp
	# create trace wvl array [m]
	leftw = cali_wvl - cali_pix*wpp
	rigthw = cali_wvl + (raw_pic.shape[1] - cali_pix)*wpp
	# those are the endpoints for the array, assert that this will generate something real (>0)
	assert(leftw>0.), 'error in calibration data, FROG did not measure negative wavelengths'
	trc_w = np.linspace(leftw,rightw,raw_pic.shape[1])
	# create trace freq array [Hz], using freq b/c python does
	# NOTE: converting spectral phase/pwr (spectrum) from wvl to freq is in Trebino pg 15
	trc_f = c/trc_w
	# NOTE: if you do this ^, trc_f wont be evenly spaced it will just be an accurate rep of what freqs were measured
	return (raw_pic,trc_d,trc_w,trc_f)

# function to call to plot the diff plots, that can be called in other functions
def nice_plot(trace,d_arr,chose_d,f_arr,chose_f,save=False,fignum=1,title='title',fname='test.png'):
	'''
	creates a nice figure of the FROG trace, I_frog(omega,tau), with plots above and to the right of that that are just I(omega) and I(tau)
	can optionally be saved as a png with dpi=300, MUST BE SAVED W COLOR OR ELSE ITS NOT A GOOD FIG
	no string formating should have to be handled on user end unless there is extra stuff you want in title for some reason
	NOTE: assuming the frequencies and delays are best displayed in THz and fs, respectively
	inputs:
	trace - array, NxM - a trace array of FROG intensities, I(omega_i,tau_j)
	d_arr - array, Nx1 - array of delay points [s], of the associated trace
	chose_d - float or int - the delay value [s] that you wish to see in the I(omega,tau=chose_d) plot
	NOTE: CURRENTLY WILL NOT work if chose_d is not an element of d_arr, in the future this funct will do its best to get close, if you wish it to be exact ensure before hand
	f_arr - array, Mx1 - array of freq points [Hz] of the associated trace
	chose_f - float or int - the freq value [Hz] that you wish to see in the I(omega=chose_f,tau) plot
	NOTE: if you do not wish to use chose_d and chose_f, might I sugguest you just use the rough plot function instead
	save - bool - would you like to save this figure? default to False
	fignum - float or int - the overall figure number, in case youd like to use this for publications/presentations, default is 1
	title - str - any info that will be displayed after Fig (fignum): title
	fname - str -relative file path and file name, for where youd like to save the image and what its name is, default is as 'test.png' in the current directory
	NOTE: will handle it if you forget to put .png at the end but please dont forget
	outputs: no outputs, just prints and (optionally) saves the figure
	'''
	# check the inputs
	assert(isinstance(trace,np.ndarray)), 'input: trace must be an array'
	assert(isinstance(d_arr,np.ndarray)), 'input: d_arr must be an array'
	assert(isinstance(f_arr,np.ndarray)), 'input: f_arr must be an array'
	assert(trace.shape==(len(d_arr),len(f_arr))), 'shape mismatch btwn trace and arrays'
	assert(isinstance(chose_d),(float,int)), 'input: chose_d must be a float or int'
	assert(isinstance(chose_f),(float,int)), 'input: chose_f must be a float or int'
	assert(isinstance(fignum,(float,int))), 'input: fignum must be a float or int'
	assert(isinstance(title,str)), 'input: title must be a string'
	assert(isinstance(fname,str)), 'input: fname must be a string'
	assert(isinstance(save,bool)), 'input: save must be bool'
	# set up all the string stuff and chosen freq/delay stuff
	# MUST SET UP HOW TO HANDLE THE CHOSEN VALUES IF THEY ARENT IN THEIR ASSOCIATED ARRAYS
	figtitle = 'Fig {}: '.format(int(fignum)) + title
	if ((np.isin(chose_d,d_arr))&(np.isin(chose_f,f_arr))): # if both are my life is easy
		b_leg = '$\\tau$={d: .1f} fs'.format(d=(chose_d*s2fs)) # names are like that bc of fig labels
		c_leg = '$\omega$={c:.3f} THz'.format(c=(chose_f*h2th))
	elif (np.isin(chose_d,d_arr)):
		b_leg = '$\\tau$={d: .1f} fs'.format(d=(chose_d*s2fs))
		# bs
		raise Exception('this is currently not supported, please make chose_f a part of f_arr')
	elif (np.isin(chose_f,f_arr)):
		c_leg = '$\omega$={c:.3f} THz'.format(c=(chose_f*h2th))
		# bs
		raise Exception('this is currently not supported, please make chose_d a part of d_arr')
	else:
		# fucker made my life hard :(
		raise Exception('this feature is currently not supported, please make both chosen values part of their associated arrays')
	# actually make the figure
	fig = plt.figure(figsize=(6,6));
	gs = fig.add_gridspec(2,2,width_ratios=(7,2),height_ratios=(2,7),left=.1,right=.9,bottom=.1,top=.9,wspace=.1,hspace=.15)
	# NOTE: these are custom adjusted values, it should work for this but i dont really know how gridspec works yet, but it seemed easier that other methods
	ax = fig.add_subplot(gs[1,0]) # main plot, trace, figure A
	ax_x = fig.add_subplot(gs[0,0],sharex=ax) # top plot, const delay, figure B
	ax_y = fig.add_subplot(gs[1,1],sharey=ax) # right plot, const freq, figure C, x,y need to be y,x for this one
	ax_x.tick_params(axis='x',labelbottom=False)
	ax_y.tick_params(axis='y',labelleft=False)
	ax.pcolormesh(f_arr*h2th,d_arr*s2fs,trace,cmap='hot')
	ax.axvline(chose_f*h2th,c='w',ls='--')
	ax.axhline(chose_d*s2fs,c='w',ls='--')
	ax.set_xlabel('Frequency [THz]')
	ax.set_ylabel('Delay [fs]')
	ax.set_title('(A)',loc='left')
	ax.set_title('$I_{FROG}(\omega_i,\\tau_j)$')
	ax_x.plot(f_arr*h2th,trace[np.isin(d_arr,chose_d),:],c='k',label=b_leg)
	ax_x.legend()
	ax_x.set_ylabel('I [cts]')
	ax_x.set_title('(B)',loc='left')
	ax_y.plot(trace[:,np.isin(f_arr,chose_f)],d_arr*s2fs,c='k',label=c_leg)
	ax_y.legend()
	ax_y.set_xlabel('I [cts]')
	ax_y.set_title('(C)',loc='left')
	# shouldnt be nessecary since i have a ref to the fig, but is generally important to have save before show because at the end of show, it closes the figure
	fig.suptitle(figtitle)
		if save:
		# check to make sure the .png prefix is present
		if (fname[-4:]!='.png'):
			fname += '.png'
		with open(fname,'w') as f:
			fig.savefig(f,dpi=300)
		except:
			raise Exception('error saving picture')
	plt.show();
	return

def rough_plot(trace,d_arr,f_arr,chose_d=False,chose_f=False,title='title'):
	'''
	creates the same 3 plots as 'nice_plot' but they are all seperate and not fancily formatted
	also you have the option only just plot the trace and not the other two plots, also it does it in grey scale
	inputs:

	outputs: no outputs, just makes the plots
	'''
	# will pretty much just copy+paste lmao
	return
