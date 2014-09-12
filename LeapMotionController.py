import Leap, sys
from Leap import CircleGesture, SwipeGesture, KeyTapGesture

class WireFrameListener(Leap.Listener):
    def __init__(self, camera):
        Leap.Listener.__init__(self)
        self.camera = camera
        
    def on_init(self, controller):
        print "Initialized"

    def on_connect(self, controller):
        print "Connected"

        controller.enable_gesture(Leap.Gesture.TYPE_CIRCLE);
        controller.enable_gesture(Leap.Gesture.TYPE_SWIPE);
        controller.enable_gesture(Leap.Gesture.TYPE_KEY_TAP);

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print "Disconnected"

    def on_exit(self, controller):
        print "Exited"

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()

        for gesture in frame.gestures():
            if gesture.type == Leap.Gesture.TYPE_CIRCLE:
                circle = CircleGesture(gesture)

                # Determine clock direction using the angle between the pointable and the circle normal
                if circle.pointable.direction.angle_to(circle.normal) <= Leap.PI/4:
                    clockwiseness = "clockwise"
                    movement = 0.1
                else:
                    clockwiseness = "counterclockwise"
                    movement = -0.1

                

                print "Rotate all in " + clockwiseness + " motion"
                self.camera.rotateAll("Z", movement)

            if gesture.type == Leap.Gesture.TYPE_SWIPE:
                swipe = SwipeGesture(gesture)

                if swipe.direction[0] > 0:
                    direction = "right"
                    movement = 10
                else:
                    direction = "left"
                    movement = -10

                print "Translate all to the " + direction
                self.camera.translateAll("x", movement)

            if gesture.type == Leap.Gesture.TYPE_KEY_TAP:
                tap = KeyTapGesture(gesture)

                print "Rotate all downwards"
                self.camera.rotateAll("X", -0.5)
