import numpy as np
# AGX Dynamics imports
import agx
import agxPython
import agxCollide
import agxIO
import agxOSG
import agxSDK
import agxTerrain
import agxRender
import agxUtil
import agxModel
import agxSensor
import agxCable
import agxWire
import os

from agxPythonModules.utils.environment import simulation, root, application, init_app
from agxPythonModules.utils.callbacks import StepEventCallback as Sec
from agxPythonModules.utils.callbacks import KeyboardCallback as Input
from agxPythonModules.utils.numpy_utils import wrap_vector_as_numpy_array

import math

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ship_shape_name = os.path.join(_THIS_DIR, "..", "..", "assets", "Gunnerus.obj")


def createTrimesh(filename: str, scale) -> agxCollide.Trimesh:
  """
  Create a trimesh from a specified mesh file
  """
  mesh = agxUtil.createTrimesh(
    filename,
    agxCollide.Trimesh.REMOVE_DUPLICATE_VERTICES,
    agx.Matrix3x3(agx.Vec3(scale)))
  assert mesh
  return mesh

class Ship(agxSDK.Assembly):
  def __init__(self,
                mass_kg=350_000.0,
                cm_shift_x=-0.2,
                thruster_z_offset=-2.5,
                thr_port_x=-10.0,
                thr_port_y=+2.76,
                thr_star_x=-10.0,
                thr_star_y=-2.76,
                thr_bow_x=12.0,
                thr_bow_y=0.0,
                shipColor=agxRender.Color.LightYellow(),
                half_length=None,
                half_width=None,
                half_height=None,
                stern_x_offset=None,
                color=None):
    super().__init__()
    
    # Create the ship
    ship_material = agx.Material("steel")
    
    ship_shape = createTrimesh(ship_shape_name, 1.0)
    assert (ship_shape)
    ship_geom = agxCollide.Geometry(ship_shape)
    ship_geom.setMaterial(ship_material)
    
    self.ship_body = agx.RigidBody(ship_geom)
    self.ship_body.getMassProperties().setMass(mass_kg) # 350t
    #ship_body.setMotionControl(agx.RigidBody.STATIC)
    
    self.ship_body.setPosition(agx.Vec3(0, 0, 0))
    
    # Mesh-aligment rotation
    self._mesh_rot = agx.EulerAngles(math.radians(90), 0, math.radians(-90))
    self._mesh_quat = agx.Quat(self._mesh_rot)
    self._mesh_quat_inv = self._mesh_quat.inverse()
    
    # Rotate 90 degrees around X and -90 around Z
    self.ship_body.setRotation(self._mesh_rot)
    self.add(self.ship_body)
    
    #Observer frame for reference
    self.f_observer = agx.ObserverFrame("ship observer", self.ship_body,
                                        agx.AffineMatrix4x4.translate(agx.Vec3(1, 0, 0)))
    #f_ob = simulation().getObserverFrame("ship observer")
    self.add(self.f_observer)
    
    # Set Center of Mass shift by moving the visual geometry relative to the body frame
    self.ship_body.getCmFrame().setLocalTranslate(agx.Vec3(cm_shift_x, 0, 0))
    
    # Thruster positions: ship-logical frame (for controller geometry)
    # and body-local frame (for AGX force application)
    self.thruster_port_local = agx.Vec3(thr_port_x, thr_port_y, thruster_z_offset)
    self.thruster_star_local = agx.Vec3(thr_star_x, thr_star_y, thruster_z_offset)
    self.thruster_bow_local = agx.Vec3(thr_bow_x, thr_bow_y, thruster_z_offset)
    self.thruster_port_body = self._mesh_quat_inv * self.thruster_port_local
    self.thruster_star_body = self._mesh_quat_inv * self.thruster_star_local
    self.thruster_bow_body = self._mesh_quat_inv * self.thruster_bow_local
    
    # Storing the hull reference
    self.hull = self.ship_body
    
    agxOSG.setDiffuseColor(agxOSG.createVisual(self.ship_body, root()), shipColor)
    
    body_fwd_candidate = agx.Vec3(0, -1, 0)
    world_fwd_at_rest = self._mesh_quat * body_fwd_candidate
    print(f"  body_fwd=[0,-1,0] -> world_fwd_at_rest="
          f"[{world_fwd_at_rest.x():.3f}, {world_fwd_at_rest.y():.3f}, {world_fwd_at_rest.z():.3f}]")
    
    # Verify at construction:
    p0 = self.ship_body.getPosition()
    p1 = self.f_observer.getPosition()
    dx = float(p1.x()) - float(p0.x())
    dy = float(p1.y()) - float(p0.y())
    psi0 = math.atan2(dy, dx)
    print(f"Ship created | observer-based heading at rest: psi0={math.degrees(psi0):.1f}°")
    print(f"  p0=[{p0.x():.3f},{p0.y():.3f},{p0.z():.3f}]")
    print(f"  p1=[{p1.x():.3f},{p1.y():.3f},{p1.z():.3f}]")
    print(f"  dx={dx:.3f}, dy={dy:.3f}")

    self.f_bow = agx.ObserverFrame("bow marker", self.ship_body,
                                       agx.AffineMatrix4x4.translate(agx.Vec3(0, -1, 0)))
    self.add(self.f_bow)

    p_bow = self.f_bow.getPosition()
    dx_bow = float(p_bow.x()) - float(p0.x())
    dy_bow = float(p_bow.y()) - float(p0.y())
    psi0_bow = math.atan2(dy_bow, dx_bow)
    print(f"  bow marker: [{p_bow.x():.3f},{p_bow.y():.3f},{p_bow.z():.3f}]")
    print(f"  bow-based heading: psi0_bow={math.degrees(psi0_bow):.1f}°")

    for label, bv in [("[1,0,0]", agx.Vec3(1,0,0)),
                          ("[0,1,0]", agx.Vec3(0,1,0)),
                          ("[0,0,1]", agx.Vec3(0,0,1)),
                          ("[0,-1,0]", agx.Vec3(0,-1,0)),
                          ("[-1,0,0]", agx.Vec3(-1,0,0)),
                          ("[0,0,-1]", agx.Vec3(0,0,-1))]:
        wv = self._mesh_quat * bv
        print(f"  mesh_quat * {label} = [{wv.x():.3f}, {wv.y():.3f}, {wv.z():.3f}]")
  def get_ship_observer(self):
    return self.f_observer.getPosition()
  
  def get_world_pose(self):
    T = self.ship_body.getTransform()
    return T.getTranslate(), self.ship_body.getRotation()
  
  def get_xy_psi(self):
    """
    Return (x, y, yaw) where yaw is the heading of the ship.
    Transform the known body-frame forward vector to world frame.
    """
    p = self.ship_body.getPosition()
    x, y = float(p.x()), float(p.y())

    q = self.ship_body.getRotation()
    fwd = q * agx.Vec3(1, 0, 0)
    fx, fy = float(fwd.x()), float(fwd.y())

    raw_yaw = math.atan2(fy, fx)
    yaw = raw_yaw + math.pi / 2.0
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    
    return x, y, yaw

  def ship_force_to_world(self, fx, fy):
    q = self.ship_body.getRotation()
    return q * (self._mesh_quat_inv * agx.Vec3(fx, fy, 0))

  def apply_thruster_forces(self, fx1, fy1, fx2, fy2, fy_bow=0.0):
    """
    Apply ship-logical forces at thruster points, converting through the
    mesh alignment rotation to body-local frame for AGX.
    Bow thruster is lateral-only (fy_bow).
    """
    f1_world = self.ship_force_to_world(fx1, fy1)
    self.ship_body.addForceAtLocalPosition(f1_world, self.thruster_port_body)

    f2_world = self.ship_force_to_world(fx2, fy2)
    self.ship_body.addForceAtLocalPosition(f2_world, self.thruster_star_body)

    f_bow_world = self.ship_force_to_world(0.0, fy_bow)
    self.ship_body.addForceAtLocalPosition(f_bow_world, self.thruster_bow_body)
    
class surfaceWater():
  def __init__(self, sim, wwc, length, width):
    
    self.water_hf = agxCollide.HeightField(50, 50, length, width, 20)
    self.heights = agx.RealVector(self.water_hf.getResolutionX() * self.water_hf.getResolutionY())
    for i in range(self.water_hf.getResolutionX() * self.water_hf.getResolutionY()):
      self.heights.append(0)
      
    self.waterSurfaceGeometry = agxCollide.Geometry(self.water_hf)
    self.waterSurfaceGeometry.setPosition(0, 0, 0)
    self.waterSurfaceMaterial = agx.Material("waterSurfaceMaterial")
    self.waterSurfaceGeometry.setMaterial(self.waterSurfaceMaterial)
    
    node = self.createWaterVisual(self.waterSurfaceGeometry)
    sim.add(self.waterSurfaceGeometry)
    self.waveFunction(0)
    
  def createWaterVisual(self, geom):
    node = agxOSG.createVisual(geom, root())
    
    agxOSG.setDiffuseColor(node, agx.Vec4f(0, 0.749020, 1, 1))
    agxOSG.setAmbientColor(node, agx.Vec4f(1))
    agxOSG.setSpecularColor(node, agx.Vec4f(1))
    agxOSG.setShininess(node, 120)
    agxOSG.setAlpha(node, 0.7)
    return node
  
  def waveFunction(self, t):
    np_heights = wrap_vector_as_numpy_array(self.heights, float).reshape(
      (self.water_hf.getResolutionX(), self.water_hf.getResolutionY()))
    
    jj = np.stack((np.arange(self.water_hf.getResolutionY()),) * self.water_hf.getResolutionX())
    ii = np.stack((np.arange(self.water_hf.getResolutionX()),) * self.water_hf.getResolutionY(), axis=1)
    waveHeight = 1.5
    
    np_heights[:] = waveHeight * (0.4 * np.sin(1.0 * jj + 0.6 * t) + 0.1 * np.sin(1.2 * ii + 0.6 * jj +1.45 * t))
    self.water_hf.setHeights(self.heights)
    
  def get_geom(self):
    return self.waterSurfaceGeometry


def buildScene1():
    application().getSceneDecorator().setEnableShadows(False)
    application().setEnableDebugRenderer(True)

    my_ship = Ship()
    my_ship.setPosition(0, 0, 3)
    simulation().add(my_ship)

    # Create surface water
    wwc = agxModel.WindAndWaterController()
    surf_water = surfaceWater(simulation(), wwc, 100, 100)
    wwc.addWater(surf_water.get_geom())
    simulation().add(wwc)

    Sec.preCollideCallback(lambda t: surf_water.waveFunction(t))


def buildScene():
    '''
    Entry point when this script is started with agxViewer
    '''
    buildScene1()


# Entry point when this script is started with python executable
init = init_app(name=__name__,
                scenes=[
                    (buildScene, '1')
                ],
                autoStepping=False)

TwoThrusterVessel = Ship  # Alias for compatibility with config parameters