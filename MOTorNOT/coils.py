import numpy as np
import attr
from MOTorNOT import rotate
from scipy.special import ellipeinc, ellipk
from scipy.constants import mu_0  #vacuum permeability SI
from scipy.optimize import root
import plotly.graph_objs as go
from MOTorNOT.plotting import plot_2D, plane_indices

def assembleCoil(wire_diameter, turns, R, Z0, I, axis):
    coils = []
    for t in range(turns):
        coils.append(Coil(R, Z0, 1, I, axis))
        Z0 += np.sign(Z0)*wire_diameter
    return Coils(coils)

@attr.s
class Coils:
    coils = attr.ib(default=[])
    def append(self, coil):
        self.coils.append(coil)

    def total_field(self, X, V=None):
        ''' Compute the total field of all coils in T '''
        #print("X.shape=",X.shape)
        #print(np.zeros(X.shape))
        #exit()
        field = np.zeros(X.shape)
        for coil in self.coils:
            field += coil.field(X, V)
        return field

    #def plot(self):
    #    from MOTorNOT.plotting import subplots
    #    subplots(self.field, numpoints=plot_params['numpoints'], label = 'B', units = 'G', scale = 1e4)
    def field_center(self):
        ''' Numerically locate the field strength minimum '''
        def field_strength(x):
            return self.total_field(np.atleast_2d(x))[0]
        return root(field_strength, x0=np.array([0, 0, 0]), tol=1e-4).x

    def plot(self, plane='xy', limits=[(-10e-3, 10e-3), (-10e-3, 10e-3)], numpoints=50,  quiver_scale=3e-4, component='all'):
        B=self.total_field
        fig = plot_2D(B, plane=plane, limits=limits, numpoints=numpoints, quiver=True, quiver_scale=quiver_scale, component=component, title='|B| (Tesla)')
        field_center = self.field_center()
        i, j = plane_indices(plane)
        fig.add_trace(go.Scatter(x=[field_center[i]], y=[field_center[j]], marker={'symbol': 'x', 'color': 'white', 'size': 12}))
        #fig.add_trace(go.Scatter(x=[0], y=[0], marker={'symbol': 'circle', 'color': 'white', 'size': 12}))
        fig.show()

    def gradient(self, X, axis='z'):
        ''' Evaluates the gradient at a point X along a given axis using a
            finite difference approximation. '''
        X = np.atleast_2d(X)
        dX = np.zeros(X.shape)
        dX[:, {'x': 0, 'y': 1, 'z': 2}[axis]] = 1e-4
        return (self.total_field(X+dX) - self.total_field(X-dX)) / 2e-4   # units: T/m


@attr.s
class Wire:
    ''' Creates a virtual wire.
        Args:
            xoffset (float): offset in x direction from the origin in m
            yoffset (float): offset in y direction from the origin in m
            current (float): current in A
            axis (int): 0, 1, or 2 to point the coil along the x, y, or z axis
    '''
    xoffset = attr.ib(converter=float)
    yoffset = attr.ib(converter=float)
    current = attr.ib(converter=float)
    axis = attr.ib(converter=int)
    def field(self, X, V = None):
        ''' Numerically evaluates the field for a wire placed at xoffset, yoffset from the origin along the z axis. Axes other than z are
            handled by rotating the coordinate system, solving along the symmetry axis, then rotating back.  units are Tesla. '''
        X = np.atleast_2d(X) #TT position in m
        if self.axis == 0:
            ''' Apply -90 degree rotation around y '''
            X = np.dot(rotate(1, -90), X.T).T
        elif self.axis == 1:
            ''' Apply 90 degree rotation around x '''
            X = np.dot(rotate(0, 90), X.T).T
        x = X[:,0]
        y = X[:,1]
        z = X[:,2]
        rho = np.sqrt((x-self.xoffset)**2+(y-self.yoffset)**2)
        field = np.zeros(X.shape)
        theta_field = mu_0*self.current/(2*np.pi*rho)
        #radial_field = 0
        field[:, 0] = -theta_field * div(y-self.yoffset, rho)
        field[:, 1] = theta_field * div(x-self.xoffset, rho)
        field[:, 2] = 0
        ''' Rotate to correct axis '''
        if self.axis == 0:
            return np.dot(rotate(1, 90), field.T).T
        elif self.axis == 1:
            return np.dot(rotate(0, -90), field.T).T
        return field

class Quadrupole2Dwires(Coils):
    def __init__(self, xoffset, yoffset, current, axis):
    # xoffset = attr.ib(default=0.005)
    # yoffset = attr.ib(default=0.005)
    # current = attr.ib(default=10.0)
    # axis = attr.ib(default=2)
    # def __init__(self, xoffset, yoffset, current, axis):
    #     ''' Creates four parallel wires. '''
    #     wire1 = Wire(xoffset=xoffset, yoffset=yoffset, current=current, axis=axis)
    #     wire2 = Wire(xoffset=-xoffset, yoffset=yoffset, current=0, axis=axis)
    #     wire3 = Wire(xoffset=xoffset, yoffset=-yoffset, current=0, axis=axis)
    #     wire4 = Wire(xoffset=-xoffset, yoffset=-yoffset, current=0, axis=axis)
    #     super().__init__(coils=[wire1, wire2, wire3, wire4])
        #print(xoffset, yoffset, current, axis)
        wire1 = Wire(xoffset=xoffset, yoffset=yoffset, current=current, axis=axis)
        wire2 = Wire(xoffset=-xoffset, yoffset=yoffset, current=-current, axis=axis)
        wire3 = Wire(xoffset=xoffset, yoffset=-yoffset, current=-current, axis=axis)
        wire4 = Wire(xoffset=-xoffset, yoffset=-yoffset, current=current, axis=axis)
        super().__init__(coils=[wire1, wire2, wire3, wire4])

class QuadrupoleCoils(Coils):
    def __init__(self, radius, offset, turns, current, axis, deltaI=0):
        ''' Creates a pair of coils with equal and opposite offsets and currents. '''
        coil1 = Coil(radius, offset, turns, current+deltaI/2, axis)
        coil2 = Coil(radius, -offset, turns, -current+deltaI/2, axis)
        super().__init__(coils=[coil1, coil2])

def div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.divide(a, b)
        res[b==0] = 0
    return res

@attr.s
class Coil():
    ''' Creates a virtual coil.
        Args:
            radius (float): coil radius in m
            offset (float): offset from the origin in m
            turns (int): number of turns
            current (float): current in A
            axis (int): 0, 1, or 2 to point the coil along the x, y, or z axis
    '''
    radius = attr.ib(converter=float, default=0.10)
    offset = attr.ib(converter=float, default=0.20)
    turns = attr.ib(converter=float, default=100)
    current = attr.ib(converter=float, default=10)
    axis = attr.ib(converter=int, default=2)

    def power(self, d):
        ''' Returns the power required to operate this coil as a function of the diameter d '''
        length = 2*np.pi*self.radius*self.turns
        resistivity = 1.68e-8
        resistance = resistivity*length/np.pi/(d/2)**2
        return np.abs(self.current)**2*resistance

    def onaxisfield(self,X,V=None):
        ''' use this well known and easily derivable formula for the on-axis field to check the off-axis formula
        http://hyperphysics.phy-astr.gsu.edu/hbase/magnetic/curloo.html '''
        X = np.atleast_2d(X)
        Bz=mu_0/(4*np.pi)*2*np.pi*self.radius**2*self.current/(X[:,2]**2+self.radius**2)**(3/2)
        return Bz

    def dipolefield(self,X,V=None):
        '''use this well known and easily derivable formula for the dipole far field (|X|>>radius) to check the off axis formula'''
        X = np.atleast_2d(X)
        field = np.zeros(X.shape)
        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]
        r=np.sqrt(x**2+y**2+z**2)
        rho=np.sqrt(x**2+y**2)
        theta=np.arcsin(rho/r)
        print(theta)
        radial_field=mu_0*self.current*self.turns*np.pi*self.radius**2/(4*np.pi*r**3)*2*np.cos(theta)
        theta_field=mu_0*self.current*self.turns*np.pi*self.radius**2/(4*np.pi*r**3)*np.sin(theta)
        rho_field = radial_field*np.sin(theta)+theta_field*np.cos(theta)
        field[:, 0] = rho_field * div(x, rho)
        field[:, 1] = rho_field * div(y, rho)
        field[:, 2] = radial_field*np.cos(theta)-theta_field*np.sin(theta)
        return field


    def field(self, X, V = None):
        ''' Numerically evaluates the field for a coil placed a distance self.Z0 from the origin along the axis of choice. Axes other than z are
            handled by rotating the coordinate system, solving along the symmetry axis, then rotating back.  units are Tesla. '''
        X = np.atleast_2d(X) #TT position in m
        if self.axis == 0:
            ''' Apply -90 degree rotation around y '''
            X = np.dot(rotate(1, -90), X.T).T
        elif self.axis == 1:
            ''' Apply 90 degree rotation around x '''
            X = np.dot(rotate(0, 90), X.T).T
        x = X[:,0]
        y = X[:,1]
        z = X[:,2]
     #   r = np.sqrt(x**2+y**2)

     #   field = np.zeros(X.shape)
     #   alpha = r/self.radius
     #   beta = (z-self.offset)/self.radius
     #   Q = (1+alpha)**2+beta**2
     #   print(alpha, beta,Q)
     #   m = 4*div(alpha, Q)

    #    gamma = div(z-self.offset, r)
    #    E_integral = ellipeinc(np.pi/2, m)
    #    K_integral = ellipk(m)

    #    prefactor = mu_0*self.turns*self.current/(2*np.pi*self.radius*Q)
    #    transverse_field = prefactor*gamma*((1+alpha**2+beta**2)/(Q-4*alpha)*E_integral-K_integral)
    #    axial_field = prefactor*((1-alpha**2-beta**2)/(Q-4*alpha)*E_integral+K_integral)

        # TT field from original distribution seems wrong.  does not agree with on-axis formula.
        # new formula taken from 1987 Metcalf paper https://doi.org/10.1103/PhysRevA.35.1535
        # and NASA01 Youngquist simple analytic expressions for the magnetic field of a circular current loop
        # and Meyrath04 UTAustin electromagnetic design basics for cold atom experiments
        #    but he seems to have a different sign in the (z-D)^2 term of the numerator of the E(k^2) term of the radial field.  he quotes Metcalf, so this is likely an error
        rho = np.sqrt(x**2+y**2)
        field = np.zeros(X.shape)
        R=self.radius
        D=self.offset
        alpha=(R+rho)**2+(z-D)**2
        beta=(R-rho)**2+(z-D)**2
        m=4*R*rho/alpha
        K_integral = ellipk(m)
        E_integral = ellipeinc(np.pi/2,m)
        prefactor = mu_0*self.turns*self.current/(2*np.pi)
        axial_field = prefactor/np.sqrt(alpha)*(K_integral+(R**2-rho**2-(z-D)**2)/beta*E_integral)
        radial_field = prefactor *div((z-D),rho)/ np.sqrt(alpha) * (-K_integral + (R ** 2 + rho ** 2 + (z - D) ** 2) / beta * E_integral)
        field[:, 0] = radial_field * div(x, rho)
        field[:, 1] = radial_field * div(y, rho)
        field[:, 2] = axial_field

        ''' Rotate to correct axis '''
        if self.axis == 0:
            return np.dot(rotate(1, 90), field.T).T
        elif self.axis == 1:
            return np.dot(rotate(0, -90), field.T).T
        return field
@attr.s
class LinearQuadrupole(Coil):
    B0 = attr.ib(converter=float, default=1)
    coils = attr.ib(default=None)
    def field(self, X, V=None):
        return np.multiply(X, np.array([1, 1, -2])) * self.B0

@attr.s
class Linear2DQuadrupole(Coil):
    B0 = attr.ib(converter=float, default=1) # gradient in T/m.  1 = 100G/cm
    #coils = attr.ib(default=None)
    def field(self, X, V=None):
        return np.multiply(X, np.array([1, 1, 0])) * self.B0
