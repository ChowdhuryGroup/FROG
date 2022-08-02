# Image processing programs for SHG FROG data
# By Adam Fisher, est 7/11/22
# NOTE: whne refering to Trebino's book I mean: Frequency-Resolved Optical Grating (2000)

# packages/boilerplate
import numpy as np 
import cv2 as cv
import scipy.interpolate as intp
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True)
mpl.rc('ytick', direction='in', right=True)
mpl.rc('xtick.minor', visible=True)
mpl.rc('ytick.minor', visible=True)

# constants
c = 2.99e8 # speed of light in vac [m/s]
# factors for plotting, b/c im working in m and sec
h2th = 1.e-15 # Hz to PHz
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
	with open(fp,'r') as f:
		raw_pic = cv.imread(fp,cv.IMREAD_UNCHANGED)
	# create trace arrays, dim dont matter until we have to crop and such
	# delay is vert, wvl is horz, pic dim is vert pix x horz pix
	# create trace delay array [s]
	trc_d = np.linspace((-raw_pic.shape[0]/2.),(raw_pic.shape[0]/2.),raw_pic.shape[0])*dpp
	# create trace wvl array [m]
	leftw = cali_wvl - cali_pix*wpp
	rigthw = cali_wvl + (raw_pic.shape[1] - cali_pix)*wpp
	# those are the endpoints for the array, assert that this will generate something real (>0)
	assert(leftw>0.), 'error in calibration data, FROG did not measure negative wavelengths'
	trc_w = np.linspace(leftw,rigthw,raw_pic.shape[1])
	# create trace freq array [Hz], using freq b/c python does
	# NOTE: converting spectral phase/pwr (spectrum) from wvl to freq is in Trebino pg 15
	trc_f = c/trc_w[::-1] # [Hz], but will now be in asc order
	# NOTE: if you do this ^, trc_f wont be evenly spaced it will just be an accurate rep of what freqs were measured
	return (raw_pic,trc_d,trc_w,trc_f)

# will want to use this to for the plot functions
def find_ind(arr,val):
	'''
	finds the index of arr that matches closest to val, useful for finding matching values in d_arr and f_arr in other functions
	NOTE: values in arrays will be in base SI units (m, s, Hz, etc) you must put in those values
	inputs:
	arr - array, Nx1 - array from which you wish to find the index of val from
	val - float - desired value that you would like to find in arr (or the closest it has), must have correct units
	outputs:
	ind - int - index from which you can find val in arr
	'''
	# will want an assert to ensure that val is within arr
	assert(isinstance(arr,np.ndarray)), 'arr must be an array'
	assert(isinstance(val,float)), 'val must be a float'
	# since its gotta work regardless of how arr is ordered (asc or desc)
	within = ((val<arr[0])&(val>arr[-1]))|((val>arr[0])&(val<arr[-1]))
	assert(within), 'val is not within range of arr, either fix val, or ensure arr is ordered'
	dx = arr[1] - arr[0]
	dist = val - arr[0]
	if np.all(np.isclose(dx,np.diff(arr))):
		ind = int(dist//dx)
	else:
		test = arr - val # should cross 0 somewhere, will use this
		a = len(arr) - arr[test>0.].shape[0]
		if (np.abs(val - arr[a])<np.abs(val - arr[a-1])):
			ind = a
		else:
			ind = a-1
	return ind

# function to call to plot the diff plots, that can be called in other functions
# HAVE find_ind FUNCT IMPLEMENT!!
def nice_plot(trace,d_arr,chose_d,f_arr,chose_f,save=False,fignum=1,title='title',fname='test.png'):
	'''
	creates a nice figure of the FROG trace, I_frog(omega,tau), with plots above and to the right of that that are just I(omega) and I(tau)
	can optionally be saved as a png with dpi=300, MUST BE SAVED W COLOR OR ELSE ITS NOT A GOOD FIG
	no string formating should have to be handled on user end unless there is extra stuff you want in title for some reason
	NOTE: assuming the frequencies and delays are best displayed in THz and fs, respectively
	inputs:
	trace - array, int, NxM - a trace array of FROG intensities, I(omega_i,tau_j)
	d_arr - array, float, Nx1 - array of delay points [s], of the associated trace
	chose_d - float - the delay value [s] that you wish to see in the I(omega,tau=chose_d) plot
	NOTE: CURRENTLY WILL NOT work if chose_d is not an element of d_arr, in the future this funct will do its best to get close, if you wish it to be exact ensure before hand
	f_arr - array, float Mx1 - array of freq points [Hz] of the associated trace
	chose_f - float - the freq value [Hz] that you wish to see in the I(omega=chose_f,tau) plot
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
	assert(isinstance(chose_d,(float))), 'input: chose_d must be a float or int'
	assert(isinstance(chose_f,(float))), 'input: chose_f must be a float or int'
	assert(isinstance(fignum,(float,int))), 'input: fignum must be a float or int'
	assert(isinstance(title,str)), 'input: title must be a string'
	assert(isinstance(fname,str)), 'input: fname must be a string'
	assert(isinstance(save,bool)), 'input: save must be bool'
	# set up all the string stuff and chosen freq/delay stuff
	# MUST SET UP HOW TO HANDLE THE CHOSEN VALUES IF THEY ARENT IN THEIR ASSOCIATED ARRAYS
	figtitle = 'Fig {}: '.format(int(fignum)) + title
	if ((np.isin(chose_d,d_arr))&(np.isin(chose_f,f_arr))): # if both are my life is easy
		b_leg = '$\\tau$={d: .1f} fs'.format(d=(chose_d*s2fs)) # names are like that bc of fig labels
		c_leg = '$\omega$={c:.3f} PHz'.format(c=(chose_f*h2th))
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
	ax.set_xlabel('Frequency [PHz]')
	ax.set_ylabel('Delay [fs]')
	ax.set_title('(A)',loc='left')
	ax.set_title('$I_{FROG}(\omega_i,\\tau_j)$')
	ax_x.plot(f_arr*h2th,trace[np.isin(d_arr,chose_d),:].T,c='k',label=b_leg)
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
		try:
			with open(fname,'w') as f:
				fig.savefig(f,dpi=300)
		except:
			raise Exception('error saving picture')
	plt.show();
	return

# HAVE find_ind FUNCT IMPLEMENT!!
def rough_plot(trace,d_arr,f_arr,chose_d=False,chose_f=False,title='title'):
	'''
	creates the same 3 plots as 'nice_plot' but they are all seperate and not fancily formatted
	also you have the option only just plot the trace and not the other two plots, also it does it in grey scale
	inputs:
	trace - array, int, NxM - a trace array of FROG intensities, I(omega_i,tau_j)
	d_arr - array, float, Nx1 - array of delay points [s], of the associated trace
	chose_d - float or int - the delay value [s] that you wish to see in the I(omega,tau=chose_d) plot
	NOTE: can leave as false if you dont want to see this plot, same for chose_f
	NOTE: CURRENTLY WILL NOT work if chose_d is not an element of d_arr, in the future this funct will do its best to get close, if you wish it to be exact ensure before hand
	f_arr - array, float, Mx1 - array of freq points [Hz] of the associated trace
	chose_f - float or int - the freq value [Hz] that you wish to see in the I(omega=chose_f,tau) plot
	NOTE: if you do not wish to use chose_d and chose_f, might I sugguest you just use the rough plot function instead
	title - str - any info that will be displayed on all 3 plots after their name (ie trace + title)
	outputs: no outputs, just makes the plots
	'''
	# check the inputs
	assert(isinstance(trace,np.ndarray)), 'input: trace must be an array'
	assert(isinstance(d_arr,np.ndarray)), 'input: d_arr must be an array'
	assert(isinstance(f_arr,np.ndarray)), 'input: f_arr must be an array'
	assert(trace.shape==(len(d_arr),len(f_arr))), 'shape mismatch btwn trace and arrays'
	assert(isinstance(chose_d,(float,bool))), 'input: chose_d must be a float/int or False'
	assert(isinstance(chose_f,(float,bool))), 'input: chose_f must be a float/int or False'
	assert(isinstance(title,str)), 'input: title must be a string'
	# do the plot main plot
	plt.figure();
	plt.pcolormesh(f_arr*h2th,d_arr*s2fs,trace,cmap='hot');
	if (isinstance(chose_d,float))&(isinstance(chose_f,float)):
		plt.axvline(chose_f*h2th,c='w',ls='--');
		plt.axhline(chose_d*s2fs,c='w',ls='--');
	elif (isinstance(chose_d,float))&(chose_f==False):
		plt.axhline(chose_d*s2fs,c='w',ls='--');
	elif (isinstance(chose_f,float))&(chose_d==False):
		plt.axvline(chose_f*h2th,c='w',ls='--');
	plt.xlabel('Frequency [PHz]');
	plt.ylabel('Delay [fs]');
	plt.title('Trace: '+title);
	# set up all the string stuff and chosen freq/delay stuff
	# MUST SET UP HOW TO HANDLE THE CHOSEN VALUES IF THEY ARENT IN THEIR ASSOCIATED ARRAYS
	if (isinstance(chose_d,(float)))&(isinstance(chose_f,(float))):
		if ((np.isin(chose_d,d_arr))&(np.isin(chose_f,f_arr))): # if both are my life is easy
			b_leg = '$\\tau$={d: .1f} fs'.format(d=(chose_d*s2fs)) # names are like that bc of fig labels
			c_leg = '$\omega$={c:.3f} PHz'.format(c=(chose_f*h2th))
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
		# then plot
		plt.figure();
		plt.plot(f_arr*h2th,trace[np.isin(d_arr,chose_d),:],c='k');
		plt.xlabel('Frequency [PHz]');
		plt.ylabel('Intensity [cts]');
		plt.title('I($\omega$) '+b_leg+title)
		plt.figure();
		plt.plot(d_arr*s2fs,trace[:,np.isin(f_arr,chose_f)],c='k');
		plt.xlabel('Delay [fs]');
		plt.ylabel('Intensity [cts]');
		plt.title('I($\\tau$) '+c_leg+title);
	elif (isinstance(chose_d,(float)))&(chose_f==False):
		if (np.isin(chose_d,d_arr)):
			b_leg = '$\\tau$={d: .1f} fs'.format(d=(chose_d*s2fs))
		else:
			# bs
			raise Exception('this is currently not supported, please make chose_d a part of d_arr')
		# then plot
		plt.figure();
		plt.plot(f_arr*h2th,trace[np.isin(d_arr,chose_d),:],c='k');
		plt.xlabel('Frequency [THz]');
		plt.ylabel('Intensity [cts]');
		plt.title('I($\omega$) '+b_leg+title)
	elif (isinstance(chose_f,(float)))&(chose_d==False):
		if (np.isin(chose_f,f_arr)):
			c_leg = '$\omega$={c:.3f} PHz'.format(c=(chose_f*h2th))
		else:
			# bs
			raise Exception('this is currently not supported, please make chose_d a part of d_arr')
		# then plot
		plt.figure();
		plt.plot(d_arr*s2fs,trace[:,np.isin(f_arr,chose_f)],c='k');
		plt.xlabel('Delay [fs]');
		plt.ylabel('Intensity [cts]');
		plt.title('I($\\tau$) '+c_leg+title);
	return

# want to remove noise then crop/change delay/freq values
# will add more filtering algorithms as time goes on
# https://towardsdatascience.com/image-filters-in-python-26ee938e57d2#:~:text=1.-,Mean%20Filter,the%20edges%20of%20the%20image.
# ^ dont actually do median filtering, even if you remove signal from filtering
# really only need diff filtering methods if getting higher noise, and will use Trebinos ideas

# but for starters: median filtering
def get_avg(trace,d_arr,f_arr,bnd_val):
	'''
	finds the median of data remaining in trace once the signal has been taken out, for the purpose of noise removal
	within its own funct in order to be as memory efficent as possible
	input:
	trace - array, int, NxM - a trace array of FROG intensities, I(omega_i,tau_j)
	d_arr - array, float, Nx1 - array of delay points [s], of the associated trace
	f_arr - array, float, Mx1 - array of freq points [Hz], of the associated trace
	bnd_val - tuple, float - tuple containing the 4 values,  ([s],[s],[Hz],[Hz]), low then high, that enclose the signal in the frog trace
	NOTE: values within bnd_val do NOT need to be actual values within d_arr and f_arr, HOWEVER, this funct uses find_ind funct, so must follow those rules
	'''
	# assertions
	assert(isinstance(trace,np.ndarray)), 'input: trace must be an array'
	assert(isinstance(d_arr,np.ndarray)), 'input: d_arr must be an array'
	assert(isinstance(f_arr,np.ndarray)), 'input: f_arr must be an array'
	assert(trace.shape==(len(d_arr),len(f_arr))), 'shape mismatch btwn trace and arrays'
	assert(isinstance(bnd_val,tuple)), 'input: bnd_val must be a tuple'
	assert(len(bnd_val)==4), 'size error for bnd_val'
	# do index array for each guy
	bnd_ind_d = np.array([find_ind(d_arr,bnd_val[0]),find_ind(d_arr,bnd_val[1])])
	bnd_ind_f = np.array([find_ind(f_arr,bnd_val[2]),find_ind(f_arr,bnd_val[3])]) # indices of interest
	# do an if to ensure that both guys are in order
	if (bnd_ind_d[0]>bnd_ind_d[1]):
		bnd_ind_d = bnd_ind_d[::-1]
	if (bnd_ind_f[0]>bnd_ind_f[1]):
		bnd_ind_f = bnd_ind_f[::-1]
	# fun fact, python passes objects into the functions so if you pass in a mutable object defined globally into the function and modify it, (as long as its not a redefinition)
	# it will be keep that change, however this only works for mutable objects :(
	zone_avg = np.zeros(4,dtype=int)
	zone_avg[:2] = np.array([np.mean(np.hsplit(trace,bnd_ind_f)[::2][i]) for i in range(2)],dtype=int)
	zone_avg[2:] = np.array([np.mean(np.vsplit(trace,bnd_ind_d)[::2][i]) for i in range(2)],dtype=int)
	# np.splits returns a view of a list of arrays so thats why the wacky indexing and list comprehension are required
	# also this only works for removing a block but np.split will be useful for the boxcar avg
	avg = int(np.mean(zone_avg))
	return avg

# put it all together
def avg_removal(trace,d_arr,f_arr,bnd_val,trace_type=np.ushort,copy=True):
	'''
	function that finds the mean of what should be signalless places of the trace and subtracting it from the whole trace to remove noise and returns a new filtered trace
	THIS IS NOT MEAN FILTERING, that is something that would distort the signal and is bad for FROG
	NOTE: ill say it here and below, the dtype of trace may be an unsigned integer, this handles that but please read dtype documentation, see below
	https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.uintc
	NOTE: uses get_med and find_ind functs within, read documentation if errors occur within said functs
	input:
	trace - array, int, NxM - a trace array of FROG intensities, I(omega_i,tau_j)
	NOTE: currently (7/27/22) i have only seen trace use unsigned 16-bit int (uint16), this funct handles the potential overflow error
	# ^HOWEVER the user must ensure that if trace is an int type that is NOT uint16, they find the correct type in numpy documentation (see above)
	d_arr - array, float, Nx1 - array of delay points [s], of the associated trace
	f_arr - array, float, Mx1 - array of freq points [Hz], of the associated trace
	bnd_val - tuple, float - tuple containing the 4 values,  ([s],[s],[Hz],[Hz]), low then high, that enclose the signal in the frog trace
	NOTE: values within bnd_val do NOT need to be actual values within d_arr and f_arr, HOWEVER, this funct uses find_ind funct, so must follow those rules
	trace_type - numpy scalar class -  default uint16 aka np.ushort, the kind of integers trace will be output with, should be what trace has as well
	NOTE: MUST use numpy scalar class names, ie np.ushort, see documentation above for types
	NOTE: currently (7/27/22) only unsigned integers are supported, i dont know how this would even work with signed integers so availiblity is tbd
	copy - bool - default True, if mem allocation is a concern (why tho) can set to false so the input trace is edited and returned rather than a copy
	NOTE: this is refering to how if a global variable that is a mutable object is passed into a function it can be modified, #iykyk, if not, leave as is
	output:
	filtered_trace - array, int, NxM - the filtered trace, can either be a copy of input trace or just modified
	'''
	# check inputs, all the basic ones are covered in get_med, so they wont be repeated here
	assert(isinstance(trace[1,1],np.unsignedinteger)), 'trace elements must be unsigned integers'
	assert(isinstance(trace[1,1],trace_type)), 'dtype mismatch btwn trace elements and trace_type'
	assert(np.issubdtype(trace_type,np.unsignedinteger)), 'output trace must be unsigned integer'
	assert(isinstance(copy,bool)), 'input: copy must be a bool'
	avg = get_avg(trace,d_arr,f_arr,bnd_val)
	# if statement should handle if trace is unsigned or not
	if copy:
		# type chnage is for unsigned ints overflow
		filtered_trace = trace.astype(float)
		filtered_trace -= avg
		filtered_trace[filtered_trace<0.] = 0.
		assert(np.all(filtered_trace>=0)) # ensure all values are >=0 before returning to int dtype
		filtered_trace = filtered_trace.astype(trace_type,copy=False) # no need for another copy ya know
		return filtered_trace
	else:
		trace = trace.astype(float,copy=False) # this will change the input trace, make ur peace with that
		trace -= avg
		trace[trace<0.] = 0.
		assert(np.all(trace>=0))
		trace = trace.astype(trace_type,copy=False)
		return trace # not 100% sure i need this return but oh well

# want crop and point adjustment as sep funct both with plot at the end so that we can see effects seperately
def man_crop(trace,d_arr,f_arr,crp_val,plot=False,chose_f=False,chose_d=False):
	'''
	function to take in a trace, its assoc arrays, and crop parameters and return a new cropped trace and arrays and optionally plot the crop
	the optional plotting just uses rough plot, you can do all 3 plots but not choose a title
	the output trace/arrays are just sections of the original trace/arrays, funct DOESNT change the points, that is a seperate function
	inputs:
	trace - array, int, NxM - a trace array of FROG intensities, I(omega_i,tau_j)
	d_arr - array, float, Nx1 - array of delay points [s], of the associated trace
	f_arr - array, float, Mx1 - array of freq points [Hz], of the associated trace
	crp_val - tuple, floats - tuple containing the 4 values,  ([s],[s],[Hz],[Hz]), that enclose the signal and become the edges of the new trace
	NOTE: crp_val must have that order of units (delay, delay, freq, freq) or will throw error
	plot - bool - default False, change to True if you wish to call rough_plot
	NOTE: see documentation for rough_plot, also defaults to only ploting the trace
	chose_d - float or int - the delay value [s] that you wish to see in the I(omega,tau=chose_d) plot
	chose_f - float or int - the freq value [Hz] that you wish to see in the I(omega=chose_f,tau) plot
	outputs: returns a tuple in the order below
	c_trace - array, int, AxB - cropped trace of FROG intensities, issa new object, original trace should be unaffected
	cd_arr - array, float, Ax1 - cropped array of delay points [s] for the new trace
	cf_arr - array, float, Bx1 - cropped array of freq points [Hz] for the new trace
	'''
	# input assertions
	assert(isinstance(trace,np.ndarray)), 'input: trace must be an array'
	assert(isinstance(d_arr,np.ndarray)), 'input: d_arr must be an array'
	assert(isinstance(f_arr,np.ndarray)), 'input: f_arr must be an array'
	assert(trace.shape==(len(d_arr),len(f_arr))), 'shape mismatch btwn trace and arrays'
	assert(isinstance(crp_val,tuple)), 'input: crp_val must be a tuple'
	assert(len(crp_val)==4), 'size error for crp_val'
	assert(isinstance(plot,bool)), 'input: plot must be a bool'
	# use find_ind 
	crp_ind_d = np.array([find_ind(d_arr,crp_val[i]) for i in range(2)])
	crp_ind_f = np.array([find_ind(f_arr,crp_val[i+2]) for i in range(2)])
	# do an if to ensure that both guys are in order
	if (crp_ind_d[0]>crp_ind_d[1]):
		crp_ind_d = crp_ind_d[::-1]
	if (crp_ind_f[0]>crp_ind_f[1]):
		crp_ind_f = crp_ind_f[::-1]
	# preallocate arrays
	cd_arr = np.zeros(np.diff(crp_ind_d))
	cf_arr = np.zeros(np.diff(crp_ind_f))
	c_trace = np.zeros((len(cd_arr),len(cf_arr)))
	# stuff em with vals
	cd_arr = d_arr[crp_ind_d[0]:crp_ind_d[1]]
	cf_arr = f_arr[crp_ind_f[0]:crp_ind_f[1]]
	c_trace = trace[crp_ind_d[0]:crp_ind_d[1],crp_ind_f[0]:crp_ind_f[1]]
	# handle plotting
	if plot:
		rough_plot(c_trace,cd_arr,cf_arr,chose_d,chose_f)
	return c_trace,cd_arr,cf_arr

# As a practical criterion, we'll consider FROG trace data to be properly sampled when 
# the intensity of the data points at the perimeter of the FROG trace grid are <=10^-4 the peak of the trace
# want a program that can do that 
# also want to do one that can worry about sampling but that is a future prob

# need sampling, std practice is about 256x256 with equal spacing 
def sr_FWHM(trace,d_arr,f_arr,pic_dim,diag=False):
	'''
	takes in trace and assoc. arrays, and returns equal temp/freq sampling rate 
	uses Trebino's book pg 216 eqn. 10.8, see chap 10 for sampling and chap 2 for pulse width
	as funct name states, it uses FWHM to obtain these values
	NOTE: if you have a pulse with lots of lobes/wings, FWHM is NOT the pulse width quantity to use
	will be used within another funct to actually create sampled arrays
	inputs:
	trace - array, int, NxM - a trace array of FROG intensities, I(omega_i,tau_j)
	NOTE: trace should already have its background removed
	d_arr - array, float, Nx1 - array of delay points [s], of the associated trace
	f_arr - array, float, Mx1 - array of freq points [Hz], of the associated trace
	pic_dim - int - the dimensions of the new trace you would like, will be NxN
	diag - bool - default False, but if True, prints useful values like temp/spec FWHM and resultant spacing
	outputs: tuple, in order below
	dt - float - temporal spacing [s] to be used in FROG trace
	df - float - spectral spacing [Hz] to be used in FROG trace
	M - float - parameter used to determine spacing
	d_max - float - delay value [s] where max(I(tau)) along line of omega = max(I(omega))
	f_max - float - freq value [Hz] where max(I(omega)) along line of tau = max(I(tau))
	'''
	# assertions
	assert(isinstance(trace,np.ndarray)), 'input: trace must be an array'
	assert(isinstance(d_arr,np.ndarray)), 'input: d_arr must be an array'
	assert(isinstance(f_arr,np.ndarray)), 'input: f_arr must be an array'
	assert(trace.shape==(len(d_arr),len(f_arr))), 'shape mismatch btwn trace and arrays'
	assert(isinstance(pic_dim,int)), 'input: pic_dim must be an integer'
	assert(isinstance(diag,bool)), 'input: diag must be a boolean'
	# index with max value
	t_max_ind = np.max(trace,axis=1).argmax()
	f_max_ind = np.max(trace,axis=0).argmax()
	# create I(omega) and I(tau) arrays
	t_curve = trace[:,f_max_ind]
	f_curve = trace[t_max_ind,:]
	d_max = d_arr[t_max_ind]
	f_max = f_arr[f_max_ind]
	if diag:
		print('max along t_curve:',s2fs*d_max)
		print('max along f_curve:',h2th*f_max)
	# find half max for both sides, using splines because itll be easier
	t_spl = intp.InterpolatedUnivariateSpline(d_arr,(t_curve.astype(float)-(np.max(t_curve.astype(float))/2.)))
	f_spl = intp.InterpolatedUnivariateSpline(f_arr,(f_curve.astype(float)-(np.max(f_curve.astype(float))/2.)))
	t_ind = np.array([find_ind(d_arr,t_spl.roots()[i]) for i in range(len(t_spl.roots()))])
	f_ind = np.array([find_ind(f_arr,f_spl.roots()[i]) for i in range(len(f_spl.roots()))])
	# avg if there is more than 1 on ea side
	# this is gonna be weird if there are tall lobes, wouldnt recommend using 
	if diag:
		print('time indices: ',t_ind,' time vals: ',s2fs*d_arr[t_ind])
		print('freq indices: ',f_ind,' freq vals: ',h2th*f_arr[f_ind])
	if (len(t_ind)>2):
		lt = np.mean(d_arr[t_ind[t_ind<t_max_ind]])
		rt = np.mean(d_arr[t_ind[t_ind>t_max_ind]])
	else:
		lt = d_arr[t_ind.min()]
		rt = d_arr[t_ind.max()]
	if (len(f_ind)>2):
		lf = np.mean(f_arr[f_ind[f_ind<f_max_ind]])
		rf = np.mean(f_arr[f_ind[f_ind>f_max_ind]])
	else:
		lf = f_arr[f_ind.min()]
		rf = f_arr[f_ind.max()]
	# create Dt,Df,M,dt,df
	Dt = np.abs(rt-lt) # temporal FWHM [s]
	Df = np.abs(rf-lf) # spectral FWHM [Hz]
	M = np.sqrt(Dt*pic_dim*Df) # factor Trebino uses for equal spacing btwn temp/spec pnts in FROG
	if diag:
		print('Dt = ',Dt*s2fs,'Df = ',Df*h2th)
	dt = Dt/M
	df = 1./(pic_dim*dt)
	if diag:
		print('M = ',M,'dt = ',dt*s2fs,'df = ',df*h2th)
	return (dt,df,M,d_max,f_max)

# need to test both padding during sampling and padding during FROG
# either way both will use sr_FWHM for now
# for starters try a funct that will default to not allow padding but can be alter to allow it
# this is super rough but it seems like just doing a linspace resampling will still be withing FSR limits (barely)
# so well make that one too

def auto_snc(trace,d_arr,f_arr,N,pad_trace=False,save=False,folder='./',fname='test'):
	'''
	auto sample + crop using sr_FWHM, as described in Trebinos book chap 10
	it will return the new trace and its associated arrays, and save it if youd like
	NOTE: b/c its calling sr_FWHM it will center both arrays on the max value, ie at pic_dim/2,pic_dim/2
	NOTE: you SHOULD NOT modify the trace or its assoc arrays
	NOTE: this is supposed to keep the delay sampling rate, dt, within the FSR but it is a v rough guess
	NOTE: if you wish to see diag results from sr_FWHM, use it seperately
	NOTE there are no checks to ensure that this sampling will capture the signal, use sr_FWHM
	inputs:
	trace - array, int, NxM - a trace array of FROG intensities, I(omega_i,tau_j)
	NOTE: trace should already have its background removed
	d_arr - array, float, Nx1 - array of delay points [s], of the associated trace
	f_arr - array, float, Mx1 - array of freq points [Hz], of the associated trace
	N - int - aka pic_dim, the dimensions of the new trace you would like, will be NxN
	pad_trace - bool - default False, if your new arrays are out of bounds of the originals it will throw an error, otherwise it will fill the trace with 0 for the out of bounds values
	save - bool - default False, whether or not the new trace/arrays will be saved
	folder - str - default current directory, relative path to directory you would like the file saved to (str)
	fname - str - extra words youd like in front of every saved file (str), ie 'first_stage'
	NOTE: do NOT include any file suffix or underscores at the end, the program will handle that for you
	outputs: if save=True will save 3 .txt files, one for the trace and each array
	NOTE: encoding: utf-8, comma delimited, and trace file will be saved as ints
	f_trace - array, int, NxM - new trace of FROG intensities
	fd_arr - array, float, Nx1 - new array of delay points [s]
	ff_arr - array, float, Mx1 - new array of freq points [Hz]
	NOTE: f is just for final
	'''
	# assert
	# call sr_FWHM
	dt,df,_,d_max,f_max = sr_FWHM(trace,d_arr,f_arr,N) # dont need M or diag messages
	# create arrays
	f_trace = np.zeros((N,N),dtype=trace.dtype)
	fd_arr = np.zeros(N)
	ff_arr = np.zeros(N)
	# center arr on max and fill on both sides
	fd_arr[N//2] = d_max
	ff_arr[N//2] = f_max
	fd_arr[:N//2] = np.array([d_max-dt*(i+1) for i in range(N//2)])[::-1]
	fd_arr[(N//2 + 1):] = np.array([d_max+dt*(i+1) for i in range(N//2 - 1)])
	ff_arr[:N//2] = np.array([f_max-df*(i+1) for i in range(N//2)])[::-1]
	ff_arr[(N//2 + 1):] = np.array([f_max+df*(i+1) for i in range(N//2 - 1)])
	# pull an assert errror if not padding trace and new arrays are out of bounds
	n2p = False # need to pad (n2p) if it becomes true will either pull an assert error or signal padding is gonna need to happen
	# create indexing for padding or just set as 1st and last index of final arrays
	if (fd_arr[0]<d_arr[0]):
		n2p = True
		ld = find_ind(fd_arr,d_arr[0])
	else:
		ld = 0
	if (fd_arr[-1]>d_arr[-1]):
		n2p = True
		rd = find_ind(fd_arr,d_arr[-1])
	else:
		rd = len(fd_arr)
	if (ff_arr[0]<f_arr[0]):
		n2p = True
		lf = find_ind(ff_arr,f_arr[0])
	else:
		lf = 0
	if (ff_arr[-1]>f_arr[-1]):
		n2p = True
		rf = find_ind(ff_arr,f_arr[-1])
	else:
		rf = len(ff_arr)
	if (pad_trace==False):
		assert(not(n2p)), 'new arrays would be out of bounds from old arrays, turn on pad_trace or adjust N'
	# use spline to fill trace values 
	trc_spl = intp.RectBivariateSpline(d_arr,f_arr,trace)
	# now to fill f_trace, need ind of where needs to be padded or not
	# fill values then pad with zeros to overwrite
	f_trace = trc_spl(fd_arr,ff_arr).astype(int) # arrays must be able to form a grid
	# also for some reason it will output neg vals, so well put all neg val -> 0
	f_trace[f_trace<0] = 0
	# now deal w/ padding
	f_trace[:ld,:] = 0
	f_trace[rd:,:] = 0
	f_trace[:,:lf] = 0
	f_trace[:,rf:] = 0
	# save if wanted
	# will overwrite if name is the same
	if save:
		trc_name = folder+fname+'_processed_trace'
		d_name = folder+fname+'_processed_delay'
		f_name = folder+fname+'_processed_freq'
		try:
			with open(trc_name,'w',encoding='utf-8') as f:
				np.savetxt(f,f_trace,fmt='%u',delimiter=',',encoding='utf-8')
		except:
			raise Exception('error saving trace file')
		try:
			with open(d_name,'w',encoding='utf-8') as f:
				np.savetxt(f,fd_arr,delimiter=',',encoding='utf-8')
		except:
			raise Exception('error saving delay array file')
		try:
			with open(f_name,'w',encoding='utf-8') as f:
				np.savetxt(f,ff_arr,delimiter=',',encoding='utf-8')
		except:
			raise Exception('error saving freq array file')
	return (f_trace,fd_arr,ff_arr)