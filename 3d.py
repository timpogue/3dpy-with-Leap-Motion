import pygame
import math
import numpy
import Leap, sys
import LeapMotionController

def square((x, z), (w, d)):
    return [[nx, nz] for nx in (x, x + w) for nz in (z, z + d)]

def circle((cx, cy, cz), (cr, res)):

    circle = Wireframe()

    theta = 0
    step = 2 * numpy.pi / res

    while theta < 2 * numpy.pi: 
        x = cx + cr * numpy.cos(theta)
        y = cy - cr * numpy.sin(theta)

        circle.addNodes([(x, y, cz)])

        theta += step

    circle.addFaces([range(res)])

    return circle

def cylinder((x, y, z), (r, h)):
    """Return a wireframe cylinder starting at (x, y, z) with a radius of r, and height of h."""
    return None

def pyramid((x, y, z), (w, h, d)):
    """Return a wireframe pyramid starting at (x, y, z) width base width w, height h, and base depth d."""

    pyramid = Wireframe()

    #First create the base
    pyramid.addNodes(numpy.array([[nx, y, nz] for nx in (x, x + w) for nz in (z, z + d)]))
    pyramid.addFaces([(0, 1, 3, 2)])

    #Find the center of the base, to do the tip at height h
    num_nodes = len(pyramid.nodes)
    meanX = sum([node[0] for node in pyramid.nodes]) / num_nodes
    meanZ = sum([node[2] for node in pyramid.nodes]) / num_nodes

    #Create top node and stitch the faces together
    pyramid.addNodes([(meanX, x - h, meanZ)])
    pyramid.addFaces([(4, 1, 0), (4, 3, 1), (4, 2, 3), (4, 0, 2)])

    return pyramid
    

def cube((x, y, z), (w, h, d)):
    """Return a wireframe cuboid starting at (x, y, z) with width, w, height, h, and depth d."""

    cube = Wireframe()
    cube.addNodes(numpy.array([[nx, ny, nz] for nx in (x, x + w) for ny in (y, y + h) for nz in (z, z + d)]))
    cube.addFaces([(0, 1, 3, 2), (7, 5, 4, 6), (4, 5, 1, 0), (2, 3, 7, 6), (0, 2, 6, 4), (5, 7, 3, 1)])   
    
    return cube

def sphere((x, y, z), (rx, ry, rz), resolution = 10):
    """Return a wireframe spheroid centered on (x, y, z) width radii of (rx, ry, rz) in respective axes."""
   
    sphere = Wireframe()

    latitudes = [n * numpy.pi / resolution for n in range(1, resolution)]
    longitudes = [n * 2 * numpy.pi / resolution for n in range(resolution)]

    # Add Nodes expect for poles
    sphere.addNodes([(x + rx * numpy.sin(n) * numpy.sin(m),
                      y - ry * numpy.cos(m),
                      z - rz * numpy.cos(n) * numpy.sin(m)) for m in latitudes for n in longitudes])
   
    # Add square faces to whole spheroid but poles
    num_nodes = resolution * (resolution - 1)
    sphere.addFaces([(m + n, (m + resolution) % num_nodes + n,
                     (m + resolution) % resolution ** 2 + (n + 1) % resolution,
                     m + (n + 1) % resolution) 
                     for n in range(resolution) for m in range(0, num_nodes - resolution, resolution)])

    # Add poles and triangular faces around poles
    sphere.addNodes([(x, y + ry, z), (x, y - ry, z)])
    sphere.addFaces([(n, (n + 1) % resolution, num_nodes + 1) for n in range(resolution)])
    start_node = num_nodes - resolution
    sphere.addFaces([(num_nodes, start_node + (n + 1) % resolution, start_node + n) for n in range(resolution)])

    return sphere

class Wireframe(object):
    def __init__(self, nodes = None):
        self.nodes = numpy.zeros((0, 4))
        self.edges = []
        self.faces = []
   
        if nodes:
            self.addNodes(nodes)

    def addNodes(self, node_array):
        ones_column = numpy.ones((len(node_array), 1))
        ones_added = numpy.hstack((node_array, ones_column))
        self.nodes = numpy.vstack((self.nodes, ones_added))

    def addEdges(self, edgeList):
        self.edges += [edge for edge in edgeList if edge not in self.edges]

    def addFaces(self, faceList, face_color = (255, 255, 255)):
        for nodeList in faceList:
            num_nodes = len(nodeList)
            if all((node < len(self.nodes) for node in nodeList)):
                self.faces.append((nodeList, numpy.array(face_color, numpy.uint8)))
                self.addEdges([(nodeList[n - 1], nodeList[n]) for n in range(num_nodes)])
    
    def sortedFaces(self):
        return sorted(self.faces, key = lambda face: min(self.nodes[f][2] for f in face[0]))

    def translate(self, axis, d):
        if axis in ["x", "y", "z"]:
            for node in self.nodes:
                if axis == "x":
                    node[0] = node[0] + d
                elif axis == "y":
                    node[1] = node[1] + d
                elif axis == "z":
                    node[2] = node[2] + d

    def scale(self, (center_x, center_y), scale):
        for node in self.nodes:
            node[0] = center_x + scale * (node[0] - center_x)
            node[1] = center_y + scale * (node[1] - center_y)
            node[2] *= scale

    def findCenter(self):

        num_nodes = len(self.nodes)
        meanX = sum([node[0] for node in self.nodes]) / num_nodes
        meanY = sum([node[1] for node in self.nodes]) / num_nodes
        meanZ = sum([node[2] for node in self.nodes]) / num_nodes

        return (meanX, meanY, meanZ)

    def rotateX(self, (cx, cy, cz), radians):
        for node in self.nodes:
            y      = node[1] - cy
            z      = node[2] - cz
            d      = math.hypot(y, z)
            theta  = math.atan2(y, z) + radians
            node[2] = cz + d * math.cos(theta)
            node[1] = cy + d * math.sin(theta)

    def rotateY(self, (cx, cy, cz), radians):
        for node in self.nodes:
            x      = node[0] - cx
            z      = node[2] - cz
            d      = math.hypot(x, z)
            theta  = math.atan2(x, z) + radians
            node[2] = cz + d * math.cos(theta)
            node[0] = cx + d * math.sin(theta)

    def rotateZ(self, (cx, cy, cz), radians):        
        for node in self.nodes:
            x      = node[0] - cx
            y      = node[1] - cy
            d      = math.hypot(y, x)
            theta  = math.atan2(y, x) + radians
            node[0] = cx + d * math.cos(theta)
            node[1] = cy + d * math.sin(theta)

    def update(self):
        pass
            
class AnimatedWireframe(Wireframe):
    def update(self):
        center = self.findCenter()
        
        self.translate("z", 10)

class Light(Wireframe):
    def __init__(self):
        super(Light, self).__init__()
        self.addNodes([[0, -1, 0]])
      
        self.min_light = 0.5
        self.max_light = 1.0
        self.light_range = self.max_light - self.min_light

class Camera:
    def __init__(self, width, height, name = "Wireframe Display"):
        self.width = width
        self.height = height

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(name)
        self.clock = pygame.time.Clock()
        self.background = (10, 10, 50)

        self.wireframes = {}

        self.perspective = False
        self.eyeX = self.width / 2
        self.eyeY = 100
        self.view_vector = numpy.array([0, 0, -100])

        self.light = Light()

        self.displayNodes = False
        self.displayEdges = False
        self.displayFaces = True
        self.nodeColor = (255, 255, 255)
        self.edgeColor = (200, 200, 200)
        self.faceColor = numpy.array((255, 255, 255))

        self.nodeRadius = 2

        self.key_to_function = {
            pygame.K_LEFT:      (lambda x: x.translateAll("x", -10)),
            pygame.K_RIGHT:     (lambda x: x.translateAll("x", 10)),
            pygame.K_DOWN:      (lambda x: x.translateAll("y", 10)),
            pygame.K_UP:        (lambda x: x.translateAll("y", -10)),
            pygame.K_EQUALS:    (lambda x: x.scaleAll(1.25)),
            pygame.K_MINUS:     (lambda x: x.scaleAll(0.8)),
            pygame.K_q:         (lambda x: x.rotateAll("X", 0.1)),
            pygame.K_w:         (lambda x: x.rotateAll("X", -0.1)),
            pygame.K_a:         (lambda x: x.rotateAll("Y", 0.1)),
            pygame.K_s:         (lambda x: x.rotateAll("Y", -0.1)),
            pygame.K_z:         (lambda x: x.rotateAll("Z", 0.1)),
            pygame.K_x:         (lambda x: x.rotateAll("Z", -0.1)),
        }

        self.control = 0

    def run(self):

        running = True
        key_down = False

        #This is where we start the leap motion controller, and add our custom listener

        listener = LeapMotionController.WireFrameListener(self)
        controller = Leap.Controller()
        controller.add_listener(listener)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    controller.remove_listener(listener)
                    pygame.quit()
                    running = False

                elif event.type == pygame.KEYDOWN:
                    key_down = event.key
                elif event.type == pygame.KEYUP:
                    key_down = None

            if key_down:
                if key_down in self.key_to_function:
                    self.key_to_function[key_down](self)

            self.clock.tick(60)
            self.display()
            pygame.display.flip()

    def addWireframe(self, name, wireframe):
        self.wireframes[name] = wireframe

    def display(self):
        self.screen.fill(self.background)
        light = self.light.nodes[0][:3]
        spectral_highlight = light + self.view_vector
        spectral_highlight /= numpy.linalg.norm(spectral_highlight)
        
        for wireframe in self.wireframes.values():

            if self.displayFaces:
                for (face, color) in wireframe.sortedFaces():
                    v1 = (wireframe.nodes[face[1]] - wireframe.nodes[face[0]])[:3]
                    v2 = (wireframe.nodes[face[2]] - wireframe.nodes[face[0]])[:3]

                    normal = numpy.cross(v1, v2)
                    towards_us = numpy.dot(normal, self.view_vector)

                    #Only draw faces that face us
                    if towards_us > 0:
                       normal /= numpy.linalg.norm(normal)
                       theta = numpy.dot(normal, light)

                       c = 0
                       if theta < 0:
                           shade = self.light.min_light * color
                       else:
                           shade = (theta * self.light.light_range + self.light.min_light) * color
                       pygame.draw.polygon(self.screen, shade, 
                                          [(wireframe.nodes[node][0], wireframe.nodes[node][1]) for node in face], 0)

            if self.displayEdges:
                for n1, n2 in wireframe.edges:
                    if self.perspective:
                        if wireframe.nodes[n1][2] > -self.perspective and wireframe.nodes[n2][2] > -self.perspective:
                            z1 = self.perspective / (self.perspective + wireframe.nodes[n1][2])
                            x1 = self.width / 2 + z1 * (wireframe.nodes[n1][0] - self.width / 2)
                            y1 = self.height / 2 + z1 * (wireframe.nodes[n1][1] - self.height / 2)

                            z2 = self.perspective / (self.perspective + wireframe.nodes[n2][2])
                            x2 = self.width / 2 + z2 * (wireframe.nodes[n2][0] - self.width / 2)
                            y2 = self.height / 2 + z2 * (wireframe.nodes[n2][1] - self.height / 2)

                            pygame.draw.aaline(self.screen, self.edgeColor, (x1, y1), (x2, y2), 1)
                    else:
                        pygame.draw.aaline(self.screen, self.edgeColor, 
                                          (wireframe.nodes[n1][0], wireframe.nodes[n1][1]),
                                          (wireframe.nodes[n2][0], wireframe.nodes[n2][1]), 1)        

            if self.displayNodes:
                for node in wireframe.nodes:
                    pygame.draw.circle(self.screen, self.nodeColor,
                                       (int(node[0]), int(node[1])), self.nodeRadius, 0)

            wireframe.update()
            self.clock.tick_busy_loop(60)

    def translateAll(self, axis, d):
        for wireframe in self.wireframes.itervalues():
            wireframe.translate(axis, d)

    def scaleAll(self, scale):
        center_x = self.width / 2
        center_y = self.height / 2

        for wireframe in self.wireframes.itervalues():
            wireframe.scale((center_x, center_y), scale)

    def rotateAll(self, axis, theta):
        rotateFunction = "rotate" + axis

        for wireframe in self.wireframes.itervalues():
            center = wireframe.findCenter()
            getattr(wireframe, rotateFunction)(center, theta)

if __name__ == "__main__":
    camera = Camera(600, 400)
    camera.perspective = 500

    resolution = 24

    sphere = sphere((200, 200, 0), (100, 100, 100), resolution)
    
    faces = sphere.faces
    for i in range(resolution / 4):
        for j in range(resolution * 2 - 4):
            f = i * (resolution * 4 - 8) + j
            faces[f][1][1] = 0
            faces[f][1][2] = 0

    camera.addWireframe("beach_ball", sphere)
    camera.run()
