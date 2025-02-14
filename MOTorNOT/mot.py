import numpy as np
import attr
from random import uniform
from scipy.constants import hbar, physical_constants # SI
from scipy.stats import poisson, norm
from scipy.optimize import root
import plotly.graph_objs as go
mu_B = physical_constants['Bohr magneton'][0]
amu = physical_constants['atomic mass constant'][0]
from MOTorNOT import load_parameters
from MOTorNOT.beams import *
atom = load_parameters()['atom']
Isat = atom['Isat'] * 10   # convert from mW/cm^2 to W/m^2.  Isat here uses the Steck definition.  Is = hbar w / 2 tau sigma.  https://steck.us/alkalidata/cesiumnumbers.1.6.pdf.
# this is half of the intensity at which stimulated emission = spont emission rate in a two-level atom
linewidth = 2*np.pi*atom['gamma'] # in rad/s
wavenumber = 2*np.pi/(atom['wavelength']*1e-9) # in m-1

@attr.s
class MOT:
    ''' This class calculates the scattering rate and optical forces arising
        from a set of beams and a magnetic field.
    '''
    beams = attr.ib(converter=list)
    field = attr.ib()
    dipolebeam = attr.ib(default = None)

    def acceleration(self, X, V):
        #print("acceleration=",self.force(X,V)/(atom['mass'] * amu))
        return self.force(X,V)/(atom['mass'] * amu)
    def acceleration_with_stochastics(self, X, V, dt=1e-5):
        return self.force_with_stochastics(X,V, dt)/(atom['mass'] * amu)

    def total_intensity(self, X):
        It = 0
        for beam in self.beams:
            It += beam.intensity(X)
        return It

    def total_absorption_rate(self, X, V):
        total_rate = 0
        for beam in self.beams:
            total_rate += self.beam_absorption_rate(beam, X, V)
        return total_rate

    def beam_absorption_rate(self, beam, X, V):
        ''' The absorption rate of the beam at a given position and velocity.
            Args:
                X (ndarray): position array with shape (N, 3)
                V (ndarray): velocity array with shape (N, 3)
                b (ndarray): magnetic field evaluated at the position
                betaT (ndarray): total saturation fraction evaluated at X
        '''
        wavevector = beam.direction * wavenumber
        prefactor = linewidth/2 * beam.intensity(X)/Isat
        summand = 0
        b = self.field(X)
        eta = self.eta(b, beam.direction, beam.handedness)
        #betaT = self.total_intensity(X)/Isat
        #betaT = beam.intensity(X)/Isat
        betaT = 0
        for mF in [-1, 0, 1]:
            amplitude = eta.T[mF+1]  # .T is array transpose.  works only with ndarray class objects
            denominator = (1+betaT+4/linewidth**2*(beam.detuning-np.dot(wavevector, V.T)-mF*atom['gF']*mu_B*np.linalg.norm(b,axis=1)/hbar)**2)
            summand += amplitude / denominator# rate = linewidth/2 * beam.intensity(X)/Isat / ( 1 + self.total_intensity(X)/Isat + (delta / gamma/2)^2 ) where delta and gamma are in radians/s  and we have taken the low intensity limit betaT=0
            # Isat here uses the Steck definition.  Is = hbar w / 2 tau sigma.  https://steck.us/alkalidata/cesiumnumbers.1.6.pdf.   this is half of the intensity at which stimulated emission = spont emission rate.
        rate = (prefactor.T*summand).T # note: this is a numpy 1d array with length = number of atoms
        return rate

    def beam_scattering_rate(self, beam, X, V):
        ''' The scattering rate of the beam at a given position and velocity.
            Args:
                 X (ndarray): position array with shape (N, 3)
                 V (ndarray): velocity array with shape (N, 3)
        '''
        return self.beam_absorption_rate(beam, X, V)/ (linewidth + 2*self.total_absorption_rate(X, V))*linewidth
        # same approach as AtomECS arXiv:2105.06447v1 eq (5)

    # def beam_scattering_rate(self, beam, X, V):
    #     ''' The scattering rate of the beam at a given position and velocity.
    #         Args:
    #             X (ndarray): position array with shape (N, 3)
    #             V (ndarray): velocity array with shape (N, 3)
    #             b (ndarray): magnetic field evaluated at the position
    #             betaT (ndarray): total saturation fraction evaluated at X
    #     '''
    #     wavevector = beam.direction * wavenumber
    #     prefactor = linewidth/2 * beam.intensity(X)/Isat
    #     summand = 0
    #     b = self.field(X)
    #     eta = self.eta(b, beam.direction, beam.handedness)
    #     betaT = self.total_intensity(X)/Isat
    #     #betaT = beam.intensity(X)/Isat
    #     for mF in [-1, 0, 1]:
    #         amplitude = eta.T[mF+1]
    #         denominator = (1+betaT+4/linewidth**2*(beam.detuning-np.dot(wavevector, V.T)-mF*atom['gF']*mu_B*np.linalg.norm(b,axis=1)/hbar)**2)
    #         summand += amplitude / denominator# rate = linewidth/2 * beam.intensity(X)/Isat / ( 1 + self.total_intensity(X)/Isat + (delta / gamma/2)^2 ) where delta and gamma are in radians/s
    #         # Isat here uses the Steck definition.  Is = hbar w / 2 tau sigma.  https://steck.us/alkalidata/cesiumnumbers.1.6.pdf.   this is half of the intensity at which stimulated emission = spont emission rate.
    #     rate = (prefactor.T*summand).T
    #     return rate

    def scattering_rate(self, X, V, i=None):
        rate = 0
        if i is not None:
            return self.beam_scattering_rate(self.beams[i], X, V)
        for beam in self.beams:
            rate += self.beam_scattering_rate(beam, X, V)
        return rate

    def force(self, X, V):
        X = np.atleast_2d(X)
        V = np.atleast_2d(V)
        force = np.atleast_2d(np.zeros(X.shape))
        #betaT = self.total_intensity(X)/Isat
        #b = self.field(X)
        wavenumber = 2*np.pi/(atom['wavelength']*1e-9)
        for beam in self.beams:
            #force += hbar* np.outer(beam.scattering_rate(X,V, b, betaT), wavenumber * beam.direction)
            force += hbar* np.outer(self.beam_scattering_rate(beam, X, V), wavenumber * beam.direction)
        if self.dipolebeam is not None:
            force += hbar*linewidth**2*self.dipolebeam.intensitygradient(X)/(8*self.dipolebeam.detuning*Isat) # from FORT equation
        return force

    def force_with_stochastics(self, X, V, dt=1e-5):
        X = np.atleast_2d(X)
        V = np.atleast_2d(V)
        force = np.atleast_2d(np.zeros(X.shape))
        wavenumber = 2*np.pi/(atom['wavelength']*1e-9) # in m-1
        #betaT = self.total_intensity(X)/Isat
        #b = self.field(X)
        #i=0
        #print("beams:", self.beams)
        #nrvs=norm.rvs
        #prvs=poisson.rvs
        #sbsr=self.beam_scattering_rate
        #hBAR=hbar
        for beam in self.beams:
            #i+=1
            #print("******************************************************************************************************************************************")
            #print("******************************************************************************************************************************************")
            #scatrate=self.beam_scattering_rate(beam, X, V)
            photons=poisson.rvs(self.beam_scattering_rate(beam, X, V)*dt)
            #photons=prvs(sbsr(beam, X, V)*dt)
            #photonrate=poisson.rvs(self.beam_scattering_rate(beam, X, V)*dt)/dt # quantized scattering rate.  1d array of length <number of atoms>
            #z=uniform(-1,1) # random point on cylinder maps to sphere.  uniform distribution on both.
            #theta=uniform(0,np.pi)
            #khat=np.array([np.sqrt(1-z**2)*np.cos(theta), np.sqrt(1-z**2)*np.sin(theta), z]) # random direction for net spontaneous force in time dt
            #force += hbar* np.outer(self.beam_scattering_rate(beam, X, V), wavenumber * beam.direction)
            absforce = hbar * np.outer(photons, wavenumber * beam.direction)/dt # absorption force with Poisson stats.  3(vector)x(number of atoms) 2Darray
            force += absforce
            #print("BEAM NUMBER",i)
            #print("beam direction=", beam.direction, "khat=",khat)
            #print("scat rate=",scatrate)
            #print("abs force=",absforce)
            #print("photonrate=",photonrate, " wavenumber=",wavenumber, " beam.direction=",beam.direction," khat=",khat)
            scale=np.sqrt(photons/3)*hbar*wavenumber/dt
            #print("scale=",scale)
            #print("rvs=",norm.rvs(scale=scale))
            #scatterforcevec=np.outer(scatterforceampl , khat)
            scatterforcevec=np.array([norm.rvs(scale=scale), norm.rvs(scale=scale), norm.rvs(scale=scale) ]).T
            force+=scatterforcevec
            #print("scatterforceampl=", scatterforceampl)
            #print("scatterforcevec=", scatterforcevec)
            #print("totalforce=",force)
        if self.dipolebeam is not None:
            force += hbar*linewidth**2*self.dipolebeam.intensitygradient(X)/(8*self.dipolebeam.detuning*Isat) # from FORT equation
        return force

    def plot(self, plane='xy', limits=[(-10e-3, 10e-3), (-10e-3, 10e-3)], numpoints=50, quiver_scale=30, component='all'):
        from MOTorNOT.plotting import plot_2D, plane_indices
        fig = plot_2D(self.acceleration, plane=plane, limits=limits, numpoints=numpoints, quiver=True, quiver_scale=quiver_scale, component=component, title='Acceleration')

        field_center = self.field_center()
        trap_center = self.trap_center()
        i, j = plane_indices(plane)
        fig.add_trace(go.Scatter(x=[field_center[i]], y=[field_center[j]], marker={'symbol': 'x', 'color': 'white', 'size': 12}))
        fig.add_trace(go.Scatter(x=[trap_center[i]], y=[trap_center[j]], marker={'symbol': 'circle-open', 'color': 'white', 'size': 12}))

        fig.show()

    def phase_plot(self, axis='x', limits=[(-10e-3, 10e-3), (-10e-3, 10e-3)], numpoints=50):
        from MOTorNOT.plotting import plot_phase_space_force, plane_indices
        import plotly.graph_objs as go
        surf = plot_phase_space_force(self.acceleration, axis=axis, limits=limits, numpoints=numpoints)
        fig = go.Figure([surf])

        fig.show()

    @staticmethod
    def eta(b, khat, s):
        ''' Transition strength to states [-1, 0, 1].
                Args:
                    b (ndarray): magnetic field, shape (N,3) array
                    khat (ndarray): beam unit vector
                    s (float): polarization handedness
            '''
        bT = b.T   # array transpose.  works only with ndarray class objects
        bnorm = np.linalg.norm(bT, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            Bhat = np.divide(bT, bnorm)
            Bhat[:, bnorm==0] = 0

        xi = khat.dot(Bhat)  # inner product i.e. cos of angle between field and wavevector
        return np.array([(1+s*xi)**2/4, (1-xi**2)/2, (1-s*xi)**2/4]).T # square of transition element to states [-1, 0, 1].  can derive by rotating field spherical tensor (+-1) into atom frame.  coherences ignored.

    def trap_center(self):
        ''' Numerically locate the potential minimum '''
        def acceleration(x):
            return self.acceleration(x, V=[0, 0, 0])[0]
        return root(acceleration, x0=[0, 0, 0], tol=1e-4).x

    def field_center(self):
        ''' Numerically locate the field strength minimum '''
        def field_strength(x):
            return self.field(np.atleast_2d(x))[0]
        return root(field_strength, x0=[0, 0, 0], tol=1e-4).x

class SixBeam(MOT):
    def __init__(self, power, radius, detuning, handedness, field, theta=0, phi=0):
        from MOTorNOT import rotate
        beams = []
        directions = [[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0]]
        for d in directions:
            d = np.dot(rotate(0, theta), d)
            d = np.dot(rotate(2, phi), d)

            beam = UniformBeam(direction = np.array(d),
                        power = power,
                        radius = radius,
                        detuning = detuning,
                        handedness = handedness)
            beams.append(beam)

        directions = [[0, 0, 1], [0, 0, -1]]
        for d in directions:
            d = np.dot(rotate(0, theta), d)
            d = np.dot(rotate(2, phi), d)
            beam = UniformBeam(direction = np.array(d),
                        power = power,
                        radius = radius,
                        detuning = detuning,
                        handedness = -handedness)
            beams.append(beam)
        super().__init__(beams, field=field)

class GratingMOT(MOT):
    def __init__(self, position, alpha, detuning, radius, power, handedness, R1, field, sectors=3, grating_radius=None, beam_type = 'uniform'):
        ''' Creates a virtual laser beam. Params dict should contain the following fields:
                position (float): grating offset from z=0
                alpha (float): diffraction angle in degrees
                radius (float): radius of incident beam
                detuning (float): detuning of incident beam
                field (method): function returning the magnetic field at a position vector X
                power (float): power of the incident beam
                handedness (float): +/-1 for circular polarization
                R1 (float): diffraction efficiency
        '''
        self.field = field
        alpha *= np.pi/180
        self.beams = []
        for n in np.linspace(1, sectors, sectors):
            self.beams.append(DiffractedBeam(n, alpha, R1*power/np.cos(alpha), radius, detuning, -handedness, position, beam_type=beam_type, sectors=sectors, grating_radius=grating_radius))
        for n in np.linspace(-1, -sectors, sectors):
            self.beams.append(DiffractedBeam(n, alpha, R1*power/np.cos(alpha), radius, detuning, -handedness, position, beam_type=beam_type, sectors=sectors, grating_radius=grating_radius))

        if grating_radius is None:
            grating_radius = 2*radius
        beam_params = {'direction': np.array([0,0,-1]),
                       'power': power,
                        'radius': radius,
                        'detuning': detuning,
                        'handedness': handedness,
                        'cutoff': grating_radius}
        if beam_type == 'uniform':
            self.beams.append(UniformBeam(**beam_params))
        elif beam_type == 'gaussian':
            self.beams.append(GaussianBeam(**beam_params))

        super().__init__(self.beams, self.field)
