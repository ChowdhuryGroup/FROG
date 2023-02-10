# imports
import numpy as np 
import scipy.interpolate as intp
import scipy.fft as fft

# functions within vanilla frog algorithm


def pulse2sig(pulse,delay):
	'''
	takes the pulse E(t) and returns the time-domain SHG signal, Esig(t,tau)
	NOTE: dont need time b/c using tau-tau will cover all the time vals
	inputs:
	pulse - 
	delay - 
	output:
	Es - 
	'''
	# asserts (later) len of pulse and delay must be same
	# just allocate space rather than for loops, plus then all the arrays are like trace already
	N = len(delay)
	# tgrid = time-delay grid [s] tgrid_ij = t_i - tau_j
	# gp = gate pulse gp_ij = E(ti-tauj) so [[E(t_i-tau)],[E(t_i+1-tau)],etc]
	# Es = SHG signal Esig(t,tau), t->omega so works the same way
	tgrid = np.array([[delay[i]-delay[j] for j in range(N)] for i in range(N)])
	real_p = intp.InterpolatedUnivariateSpline(delay,pulse.real,ext='zeros') # extrapolations = 0
	imag_p = intp.InterpolatedUnivariateSpline(delay,pulse.imag,ext='zeros')
	gp = np.array([[(real_p(tgrid[i,j]) + 1.j*imag_p(tgrid[i,j])) for j in range(N)] for i in range(N)])
	# ^ dont matter if pnts are ordered whn eval spl, but even if it did, doesnt matter like this
	Es = np.array([[pulse[i]*gp[i,j] for j in range(N)] for i in range(N)],dtype=complex)
	return Es



def FTsig(Es):
	'''
	go from Esig(t,tau) to Esig(omega,tau) via a 1-d FT
	NOTE: using ortho FT convention
	CURRENTLY (8/11/22) DOESNT handle padding with extra zeros, so i would recommend not
	input:
	Es -
	CURRENTLY NOT GIVING BACK FFT.FREQ ARRAY (10/10/22)
	dt - 
	N - 
	output: tuple in that order
	ftEs - 
	ft_freq - not the same as the processed freqs (f_arr)
	'''
	# assert, idk yet
	# NOTE: when only FT 1 axis, the shift funct also needs to know which axis to shift
	ftEs = fft.fftshift(fft.fft(Es,axis=0,norm='ortho'),axes=0)
	# ft_freq = fft.fftshift(fft.fftfreq(N,dt))
	return ftEs


def apply_frog(ftEs,m_trace):
	'''
	applies the measured FROG trace to the current iteration's generated E_sig(omega,tau) to generate the improved E'_sig(omega,tau), aka eqn 8.4 in Trebino's book
	this is the general function but within my overal FROG algorithm
	inputs:
	ftEs -
	m_trace - 
	outputs:
	ftEs_p - 
	'''
	# asserts
	# need to make a list comp becuase if any vals of ftEs are 0 its gonna throw nans
	# af = np.zeros_like(ftEs)
	af = np.zeros_like(ftEs)
	af = (ftEs/np.abs(ftEs))*np.sqrt(m_trace)
	ftEs_p = np.array([[af[i,j] if (ftEs[i,j]!=0) else 0. for j in range(af.shape[0])] for i in range(af.shape[0])])
	return ftEs_p


def sig2trc(ftEs):
	'''
	computes the kth calculated trace from the kth Es(omega,tau) for use in G error
	not partial'ed b/c ftEs changes every iteration and is only input
	can handle 1-d arrays for so this can be used for other purposes
	input:
	ftEs - 
	output:
	c_trace - 
	'''
	return np.abs(ftEs)**2


def mu_factor(m_trace,c_trace):
	'''
	calculates the scale factor, mu, for the kth iteration of the FROG algorithm
	NOTE: currently (8/15/22), as i understand, this is to be re-calculated for every iteration, this may change as my understanding changes
	inputs:
	m_trace - needs to be normalized, is done in basic_frog
	c_trace - 
	outputs:
	mu_k - 
	'''
	# assertions, trace shapes
	mu_k = np.divide(np.sum(np.multiply(m_trace,c_trace)),np.sum(np.multiply(c_trace,c_trace)))
	return mu_k


# might need to normalize calc trace 
def g_err(m_trace,c_trace):
	'''
	calculated the G error for the kth iteration of the FROG algorithm
	this quantity will be the error that defines convergence and will cause the overall program to stop once it gets low enough
	calls mu_factor within funct
	inputs:
	m_trace - needs to be normalized, is done in basic_frog
	c_trace - 
	outputs:
	Gk - 
	'''
	# assert, trace shapes (if not in mu_factor)
	# get factors
	N = m_trace.shape[0]
	mu = mu_factor(m_trace,c_trace)
	# calc gk
	# asssert that gk will stay real either before w/ stuff in sum or after w/ dtype check
	gk = np.sqrt(np.sum((m_trace - mu*c_trace)**2)/(N**2))
	return gk


def IFTsig(ftEs_p):
	'''
	go from ftEsig'(omega,tau) to Esig'(t,tau) after applying measured FROG trace, via a 1-d IFT
	NOTE: using ortho FT convention
	CURRENTLY (8/11/22) DOESNT handle padding with extra zeros, do not use like that
	input:
	ftEs_p - 
	CURRENTLY NOT GIVING BACK FFT.FREQ ARRAY (10/10/22)
	dw - 
	N - 
	output: tuple in this order
	Es_p - 
	ift_time - array, Nx1 - arr of times from IFT
	'''
	# asserts, idk yet
	# need to undo fft.fftshift b/c ift expects direct output from ft
	Es_p = fft.ifft(fft.ifftshift(ftEs_p,axes=0),axis=0,norm='ortho')
	# ift_time = fft.fftshift(fft.fftfreq(N,dw))
	return Es_p


def gen_Ekp1(Es_p):
	'''
	generates the k+1th E(t) from the kth E'(t,tau) to be used as the pulse in the k+1th FROG cycle
	this is a funct that will be utilized within main program, so it will be partialed
	inputs:
	Es_p - 
	dtau - float
	OR
	delays - array
	NOTE: using np.trapz, can use array or spacing or array, since delays is evenly spaced i wanna go w/ just having spacing since i should probs alrdy have tht
	outputs:
	Ekp1 - 
	'''
	# asserts: if array then check shapes
	# trapz reduces dim of array since it sums along an axis, since delays are horz values need axis=1
	# CURRENTLY NOT GOING TO INCLUDE DX=DTAU, b/c it makes it hella small
	# np.trapz(Es_p,dx=dt,axis=1) # or could do x=d_arr
	Ekp1 = np.trapz(Es_p,axis=1)
	return Ekp1

# actual program
# what do we need: m_trace, inital guess, max_iter, g val for converge
# eventually: dt, df, might want N

# dont apply until it works with all others
def basic_frog(m_trace,E0,delay,max_iter,min_g):
	'''
	vanilla pulse retreval algorithm

	inputs:
	m_trace - normalized? or normalize within
	E0 - inital guess
	delay - d_arr
	max_iter - 
	min_g - have default ~10^-3
	output: tuple in this order
	Ef - retrived pulse
	ret_trace - retrived trace
	G - array of g err, holds no zeros if algorithm finishes before max_iter
	'''
	# asserts
	# allocate
	mtrc = m_trace.astype(float)/m_trace.max()
	G_err = np.zeros(max_iter) # array holding g err's, for now
	N = m_trace.shape[0] # should be NxN
	Ef = np.zeros(N,dtype=complex) # final retrived E(t)
	ret_trace = np.zeros((N,N)) # trace of final iterations ftEs_p
	# counter
	k = 0
	# while loop for algorithm
	while (k<max_iter):
		# allocate all arrays (so they reset ea time) and if for E_guess 
		if (k==0):
			Ek = E0
		else:
			Ek = E_kp1
		Es = np.zeros((N,N),dtype=complex) # Esig(t,tau)
		ftEs = np.zeros((N,N),dtype=complex) # Esig(omega,tau)
		c_trace = np.zeros((N,N)) # calc Ifrog(omega,tau)
		ftEs_p = np.zeros((N,N),dtype=complex) # Esig(omega,tau) after applying measured frog
		Es_p = np.zeros((N,N),dtype=complex) # Esig'(t,tau)
		E_kp1 = np.zeros(N,dtype=complex) #Ek+1(t)
		# gen signal
		Es = pulse2sig(Ek,delay)
		# FT
		ftEs = FTsig(Es)
		# calc trace
		c_trace = sig2trc(ftEs)
		c_trace /= c_trace.max()
		# apply frog, calc gk
		ftEs_p = apply_frog(ftEs,mtrc)
		G_err[k] = g_err(mtrc,c_trace)
		Es_p = IFTsig(ftEs_p)
		E_kp1 = gen_Ekp1(Es_p)
		# check converge condits and break loop if done
		print('for iteration: ',k+1,', g = ',G_err[k]) # dont know if i want this
		if (G_err[k]<=min_g):
			print('Yay pulse retrived in ',k+1,' iterations!')
			break
		# incroment k, must be last thing
		k += 1
		# max iter reached, give the people their values
	G = np.zeros(k+1) # G error array that you get to keep
	G = G_err[:k]
	ret_trace = sig2trc(ftEs_p)
	Ef = E_kp1
	return (Ef,ret_trace,G)


def IP(E,I_lim):
	'''
	takes a complex E field and returns the normalized intensity and phase of the field
	this handles the phase wrapping and blanking needed to make a decent plot
	inputs:
	E - 
	I_lim - float - default 1.e-3, intensity limit after which youd like phase blanking to occur
	outputs: tuple in this order
	I - 
	phi - 
	'''
	# asserts, needs to be complex or handle shit that isnt
	I = sig2trc(E)
	I /= I.max()
	# well figure out if this needs the neg sign once this actually works
	ang = np.unwrap(np.angle(E)) # handles unwrapping
	phi = np.array([ang[i] if I[i]>I_lim else 0. for i in range(len(E))]) # handles blanking
	return (I,phi)