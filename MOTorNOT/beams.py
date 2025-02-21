import numpy as np
import attr
from scipy.spatial.transform import Rotation

@attr.s
class Beam:
    ''' Prototype class for describing laser beams. Subclasses for Uniform,
        Gaussian, or Diffracted beams will implement their own methods describing
        their spatial intensity variation.
    '''
    direction = attr.ib(converter=np.array) #must be a unit vector array e.g. [0,1,0]
    power = attr.ib(converter=float) #power in W
    radius = attr.ib(converter=float) #1/e^2 intensity beam radius in m
    detuning = attr.ib(converter=float)
    handedness = attr.ib(converter=int)
    origin = attr.ib(default=np.array([0, 0, 0]))
    cutoff = attr.ib(default=None) # radius at which intensity cuts to zero. in units of m except for Guidebeam, which is in units of radius.
    holedia = attr.ib(default=None)
    #stageoffset = attr.ib(default=None) # separation between beams for multiple beams per beam
    #stagenr = attr.ib(default=None) # number of beams for multiple beams per beam
    odtwavelength = attr.ib(default=None)  #TT added for dipole force trap beams
    # tetrahedron pit trap.  plane of mirror is assumed x-y.  incoming beam assumed from +z to -z.
    # origin of refl beams is at pit vertex z=-a/(2 sqrt(3)) tan theta. reflected beams go up
    # overall system origin is above this (quadrupole field null)
    pit_side = attr.ib(default=None)  # = a = side of equilateral triangle pit for tetrahedron trap triangle beams in m https://doi.org/10.1364/OE.17.013601
    pit_angle = attr.ib(default=(np.pi-np.arccos(-1/3))/2)   # = theta = triangle pit declination angle from horizontal in rad https://doi.org/10.1364/OE.17.013601

@attr.s
class UniformBeam(Beam):
    ''' Creates a virtual laser beam.
        Args:
            direction (array-like): a unit vector representing the beam's direction
            power (float)
            radius (float): radius of the beam. Beams are currently treated as uniform intensity within the radius.
            detuning (float)
            handedness (int): +/- 1 for circular polarization.
    '''
    def exists_at(self, X):
        ''' A boolean check for whether or not the beam exists at position X. Only works for beams along the x, y, or z axes; arbitrary directions will be supported later. Also assumes that the beam passes through the origin. '''
        X0 = X-self.origin
        r = np.linalg.norm(-X0+np.outer(np.dot(X0, self.direction), (self.direction)), axis=1)
        if self.cutoff is None:
            return r < self.radius
        else:
            return (r < self.radius) & (r < self.cutoff)

    def intensity(self, X):
        return self.exists_at(X) * self.power/np.pi/self.radius**2

def between_angles(theta, a, b):
    theta = theta % (2*np.pi)
    a = a % (2*np.pi)
    b = b % (2*np.pi)
    if a < b:
        return np.logical_and(a <= theta, theta <= b)
    return np.logical_or(a <= theta, theta <= b)

class DiffractedBeam(Beam):
    def __init__(self, n, alpha, power, radius, detuning, handedness, position, origin = np.array([0,0,0]), beam_type='uniform', sectors=3, grating_radius=None):
        self.beam_type = beam_type
        self.n = n
        self.sectors = sectors
        self.phi = (np.pi * (1 + (2*np.abs(n)+1)/sectors)) % (2*np.pi)
        self.direction = np.array([np.sign(n)*np.cos(self.phi)*np.sin(alpha),
                                   np.sign(n)*np.sin(self.phi)*np.sin(alpha),
                                   np.cos(alpha)])
        super().__init__(self.direction, power, radius, detuning, handedness, cutoff=grating_radius)
        self.alpha = alpha
        self.I = power/np.pi/radius**2
        self.z0 = -position
        self.grating_radius = grating_radius
        if grating_radius is None:
            self.grating_radius = radius

    def intensity(self, X):
        x = X.T[0]
        y = X.T[1]
        z = X.T[2]
        z0 = self.z0
        ## trace wavevector backwards to grating plane and calculate intensity
        x0 = x - (z-z0)*self.direction[0]/self.direction[2]
        y0 = y - (z-z0)*self.direction[1]/self.direction[2]
        r0 = np.sqrt(x0**2+y0**2)

        I = self.I
        if self.beam_type == 'gaussian':
            I *= np.exp(-2*r0**2/self.radius**2)

        ## check if the in-plane point is in the sector; return 0 if not
        radial_inequality = (r0 <= self.radius) & (r0 <= self.grating_radius)
        phi = np.mod(np.arctan2(y0, x0), 2*np.pi)
        angular_inequality = between_angles(phi, self.phi + np.pi - np.pi/self.sectors, self.phi + np.pi + np.pi/self.sectors)
        axial_inequality = z > z0

        return I * radial_inequality * angular_inequality * axial_inequality


@attr.s
class GaussianBeam(Beam):
    def intensity(self, X):
        X0 = X-self.origin# X0 is a vector of position vectors
        r = np.linalg.norm(-X0+np.outer(np.dot(X0,self.direction),(self.direction)),axis=1)#  norm axis=1 squares all elements in array then adds the elements of each row and sqrts, giving a vector.  r is a vector of radii
        w = self.radius
        ##I = self.power/np.pi/self.radius**2 TT this is wrong.  corrected.
        I = 2 * self.power / np.pi / self.radius ** 2
        return I*np.exp(-2*r**2/w**2) * (r <= self.cutoff)

@attr.s
class GaussianBeamWithHole(Beam):
    def intensity(self, X):
        X0 = X-self.origin
        r = np.linalg.norm(-X0+np.outer(np.dot(X0,self.direction),(self.direction)),axis=1)
        w = self.radius
        ##I = self.power/np.pi/self.radius**2 TT this is wrong.  corrected.
        I = 2*self.power / np.pi / self.radius ** 2
        return I*np.exp(-2*r**2/w**2) * (r <= self.cutoff) * (r >= self.holedia/2)

@attr.s
class GuideBeam(Beam): #beam is in fiber starting at the origin and going in -self.direction.  beam is in free space starting at the origin and going in +self.direction
    def intensitygradient(self, X): # X is a vector of points
        X0 = X-self.origin
        #print(X0)
        #print(X0.shape[0])
        z = np.dot(X0, self.direction)  # vector of z coordinates
        #print(z)
        zvectors=self.direction*X0
        #print(zvectors)
        rho = np.linalg.norm(X0-np.outer(np.dot(X0,self.direction),self.direction), axis=1)# vector of rho coordinates
        #print(rho)
        rhovectors = X0-np.outer(np.dot(X0,self.direction),self.direction)
        #print(rhovectors)
        #print((rhovectors.T/rho).T)
        zR=np.pi*self.radius**2/self.odtwavelength
        w = self.radius*(z<=0)+self.radius*np.sqrt(1+(z/zR)**2)*(z>0)
        #print(w)
        I0 = 2 * self.power / np.pi / self.radius ** 2
        #print(rho/w**2)
        #print( (rho / w ** 2 *(rhovectors.T/rho)).T)
        drho= (I0*(-4*rho/w**2 )*np.exp(-2*rho**2/w**2) * (rho <= self.cutoff*w) *(rhovectors.T/rho)).T
        dz= (I0*((2*z/zR**2)*(-1/(1+(z/zR)**2)**2 + 2*rho**2/w**2/(1+(z/zR)**2)**3))*np.exp(-2*rho**2/w**2) * (rho <= self.cutoff*w) *(zvectors.T/z)).T
        #drho= (I0*(-4*rho/w**2 )*np.exp(-2*rho**2/w**2)  *(rhovectors.T/rho)).T # dI/drho * rhohat#
        drho = (I0*(self.radius/w)**2 * (-4 * rho / w ** 2) * np.exp(-2 * rho ** 2 / w ** 2) * (rhovectors.T / rho)).T  # dI/drho * rhohat
        #dz= (I0*((2*z/zR**2)*(-1/(1+(z/zR)**2)**2 + 2*rho**2/w**2/(1+(z/zR)**2)**3))*np.exp(-2*rho**2/w**2)  *(zvectors.T/z)).T # dI/z * zhat
        dz= (I0*((2*z/(zR**2*(1+(z/zR)**2)**2)*(2*rho**2/self.radius**2/(1+(z/zR)**2))-1)*np.exp(-2*rho**2/w**2) *(zvectors.T / z)).T * (z>0)  # dI/z * zhat
        #print(drho)
        #print(dz)
        #print(drho+dz)
        #return I0*( (-4*rho/w**2) + (2*z/zR**2)*(-1/(1+(z/zR)**2)**2 + 2*rho**2/w**2/(1+(z/zR)**2)**3)*self.direction )*np.exp(-2*rho**2/w**2) * (rho <= self.cutoff*w)
        return drho+dz # gradient in cylindrical coords is dI/drho * rhohat + 1/rho*dI/dphi*phihat + odI/z * zhat

@attr.s
class TriangleBeamGaussian(Beam):
    def intensity(self, X):
        X0 = X-self.origin# X0 is a vector of position vectors relative to beam origin instead of field null
        r = np.linalg.norm(-X0 + np.outer(np.dot(X0, self.direction), (self.direction)), axis=1)
        axis = np.cross(np.array([0, 0, 1]), self.direction)  # rotate X0s around this axis to bring system xy plane into xy plane of beam
        axis = axis / np.linalg.norm(axis) # normalized axis
        rotation_angle = - np.arccos(np.dot(np.array([0, 0, 1]), self.direction))
        rot = Rotation.from_rotvec(rotation_angle * axis)
        newX0 = rot.apply(X0)
        newX = np.dot(newX0, axis) #component of position vector along rotation axis is newX component.  it is in plane of mirror parallel to substrate plane
        # newYvec = newX0 - np.dot(axis, newX0)*axis - np.dot(np.array([0, 0, 1]), newX0)*np.array([0, 0, 1])# newY vector in plane of mirror
        newYhat = np.cross(np.array([0, 0, 1]), axis)
        newYvec = newX0 - np.outer(np.dot(X0, newYhat), newYhat)  # newY vector in plane of mirror
        newY = np.dot( newYvec , np.cross(np.array([0,0,1]), axis) ) # y component in direction up mirror wall.  newY axis connects pit vertex to pit edge ctr
        w = self.radius
        I = 2 * self.power / np.pi / self.radius ** 2
        return I * np.exp(-2 * r ** 2 / w ** 2) * (newY>0) * (newY / np.absolute(newX) > 1 / (np.sqrt(3) * np.cos(self.pit_angle))) \
               * (np.dot(X0, np.array([0, 0, 1])) > 0) \
            * (newY < self.pit_side / (2 * np.sqrt(3) * np.cos(self.pit_angle)))
    def intensity2(self, X):
        X0 = X-self.origin# X0 is a vector of position vectors relative to beam origin
        r = np.linalg.norm(-X0 + np.outer(np.dot(X0, self.direction), (self.direction)), axis=1)
        axis = np.cross(np.array([0, 0, 1]), self.direction)  # rotate X0s around this axis to bring system xy plane into xy plane of beam
        axis = axis / np.linalg.norm(axis) # normalized axis
        rotation_angle = - np.arccos(np.dot(np.array([0, 0, 1]), self.direction))
        rot = Rotation.from_rotvec(rotation_angle * axis)
        newX0 = rot.apply(X0)
        newX = np.dot(newX0, axis) #component of position vector along rotation axis is newX component.  it is in plane of mirror parallel to substrate plane
        #newYvec = newX0 - np.dot(axis, newX0)*axis - np.dot(np.array([0, 0, 1]), newX0)*np.array([0, 0, 1])# newY vector in plane of mirror
        newYhat = np.cross(np.array([0, 0, 1]),axis)
        newYvec = newX0 - np.outer(np.dot(X0, newYhat), newYhat)  # newY vector in plane of mirror
        newY = np.dot( newYvec , np.cross(np.array([0,0,1]), axis) ) # y component in direction up mirror wall.  newY axis connects pit vertex to pit edge ctr
        w = self.radius
        I = 2 * self.power / np.pi / self.radius ** 2
        return I * np.exp(-2 * r ** 2 / w ** 2) * (newY>0) * (newY / np.absolute(newX) > 1 / (np.sqrt(3) * np.cos(self.pit_angle))) \
               * (np.dot(X0, np.array([0, 0, 1])) > 0) \
            * (newY < self.pit_side / (2 * np.sqrt(3) * np.cos(self.pit_angle))), X0, newX0, newX, newY, axis, rotation_angle*180/np.pi

class TriangleBeamUniform(Beam):
    def intensity(self, X):
        X0 = X-self.origin# X0 is a vector of position vectors relative to beam origin
        r = np.linalg.norm(-X0 + np.outer(np.dot(X0, self.direction), (self.direction)), axis=1)
        axis = np.cross(np.array([0, 0, 1]), self.direction)  # rotate X0s around this axis to bring system xy plane into xy plane of beam
        axis = axis / np.linalg.norm(axis) # normalized axis
        rotation_angle = - np.arccos(np.dot(np.array([0, 0, 1]), self.direction))
        rot = Rotation.from_rotvec(rotation_angle * axis)
        rot2 = Rotation.from_rotvec(rotation_angle / 2 * axis)
        newX0 = rot.apply(X0)
        faceX0 = rot2.apply(X0)
        newX = np.dot(newX0, axis) #component of position vector along rotation axis is newX component.  it is in plane of mirror parallel to substrate plane
        #print(newX0)
        #print(axis)
        #m=np.size(newX0,0)
        #axisvec=np.tile(axis,(m,1))
        #print('axisvec ',axisvec)
        #print(np.dot(newX0,axis))
        #print('np.dot(newX0,axisvec) ',np.dot(newX0,axisvec))
        newYhat=rot.apply(np.cross(self.direction,axis))
        newY = np.dot(newX0, newYhat)
        #newYvec = newX0 - np.dot(newX0, axisvec) * axis - np.dot(newX0, np.array([0, 0, 1])) * np.array( [0, 0, 1])  # newY vector in plane of mirror
        #newYhat = np.cross(np.array([0, 0, 1]),axis)
        #newYvec = newX0 - np.outer(np.dot(X0, newYhat), newYhat)  # newY vector in plane of mirror
        #newY = np.dot( newYvec , np.cross(np.array([0,0,1]), axis) ) # y component in direction up mirror wall.  newY axis connects pit vertex to pit edge ctr
        w = self.radius
        I = 2 * self.power / np.pi / self.radius ** 2
        return I * (r < self.cutoff) * (newY > 0)* (newY / np.absolute(newX) > 1 / (np.sqrt(3)))* (newY < self.pit_side / (2 * np.sqrt(3))) \
            * (np.dot(faceX0, np.array([0, 0, 1])) > 0)
    def intensity2(self, X):
        X0 = X-self.origin# X0 is a vector of position vectors relative to beam origin
        r = np.linalg.norm(-X0 + np.outer(np.dot(X0, self.direction), (self.direction)), axis=1)
        axis = np.cross(np.array([0, 0, 1]), self.direction)  # rotate X0s around this axis to bring system xy plane into xy plane of beam
        axis = axis / np.linalg.norm(axis) # normalized axis
        rotation_angle = - np.arccos(np.dot(np.array([0, 0, 1]), self.direction))
        rotation_angle = - (np.pi / 2 - np.arccos(np.dot(np.array([0, 0, 1]), self.direction)))
        rot = Rotation.from_rotvec(rotation_angle * axis)
        rot2 = Rotation.from_rotvec(rotation_angle/2 * axis)
        newX0 = rot.apply(X0)
        faceX0 = rot2.apply(X0)
        newX = np.dot(newX0, axis) #component of position vector along rotation axis is newX component.  it is in plane of mirror parallel to substrate plane
        #print(newX0)
        #print(axis)
        #m=np.size(newX0,0)
        #axisvec=np.tile(axis,(m,1))
        #print('axisvec ',axisvec)
        #print(np.dot(newX0,axis))
        #print('np.dot(newX0,axisvec) ',np.dot(newX0,axisvec))
        newYhat=rot.apply(np.cross(self.direction,axis))
        newY = np.dot(newX0, newYhat)
        #newYvec = newX0 - np.dot(newX0, axisvec) * axis - np.dot(newX0, np.array([0, 0, 1])) * np.array( [0, 0, 1])  # newY vector in plane of mirror
        #newYhat = np.cross(np.array([0, 0, 1]),axis)
        #newYvec = newX0 - np.outer(np.dot(X0, newYhat), newYhat)  # newY vector in plane of mirror
        #newY = np.dot( newYvec , np.cross(np.array([0,0,1]), axis) ) # y component in direction up mirror wall.  newY axis connects pit vertex to pit edge ctr
        w = self.radius
        I = 2 * self.power / np.pi / self.radius ** 2
        return I * (r < self.cutoff) * (newY > 0) * (newY / np.absolute(newX) > 1 / (np.sqrt(3))) \
               * (np.dot(faceX0, np.array([0, 0, 1])) > 0) \
               * (newY < self.pit_side / (2 * np.sqrt(3))), X0, newX0, newX, newY, axis, rotation_angle*180/np.pi