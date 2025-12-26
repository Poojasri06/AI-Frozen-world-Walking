import cv2
import numpy as np
import mediapipe as mp
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math
import random
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Player position and movement with realistic physics
player_x, player_y, player_z = 0.0, 1.6, 5.0
velocity_x, velocity_z = 0.0, 0.0
camera_yaw = 0.0  # Horizontal rotation
camera_pitch = 0.0  # Vertical rotation (looking up/down)
target_yaw = 0.0
target_pitch = 0.0

# Movement parameters
max_speed = 0.25
acceleration = 0.015
friction = 0.92
rotation_speed = 2.0

# Expanded frozen world objects
class IceCrystal:
    def __init__(self, x=None, z=None):
        self.x = x if x is not None else random.uniform(-200, 200)
        self.z = z if z is not None else random.uniform(-200, 200)
        self.y = random.uniform(0.5, 2.0)
        self.size = random.uniform(0.3, 0.8)
        self.height = random.uniform(1.5, 3.0)
        self.rotation = random.uniform(0, 360)
        self.rotation_speed = random.uniform(0.2, 0.5)
        self.offset = random.uniform(0, 2 * math.pi)
        self.float_speed = random.uniform(0.5, 1.5)

class FrozenTree:
    def __init__(self, x=None, z=None):
        self.x = x if x is not None else random.uniform(-200, 200)
        self.z = z if z is not None else random.uniform(-200, 200)
        self.y = 0.0
        self.height = random.uniform(5, 12)
        self.trunk_radius = random.uniform(0.3, 0.6)
        self.foliage_radius = random.uniform(2.0, 4.0)

class Snowflake:
    def __init__(self):
        self.x = random.uniform(-250, 250)
        self.y = random.uniform(0, 80)
        self.z = random.uniform(-250, 250)
        self.speed = random.uniform(0.02, 0.05)
        self.drift_x = random.uniform(-0.01, 0.01)
        self.drift_z = random.uniform(-0.01, 0.01)
        self.size = random.uniform(0.03, 0.08)

class IceBoulder:
    def __init__(self):
        self.x = random.uniform(-200, 200)
        self.z = random.uniform(-200, 200)
        self.y = random.uniform(0.5, 1.5)
        self.size = random.uniform(1.0, 3.0)

# Create massive frozen world with dynamic spawning
ice_crystals = [IceCrystal() for _ in range(150)]
frozen_trees = [FrozenTree() for _ in range(100)]
snowflakes = [Snowflake() for _ in range(2000)]
ice_boulders = [IceBoulder() for _ in range(50)]

# Previous pose tracking for movement detection
prev_left_knee_y = None
prev_right_knee_y = None
step_threshold = 0.03
is_walking = False

def calculate_distance(p1, p2):
    """Calculate distance between two points"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def process_pose(frame):
    """Advanced pose processing for realistic movement control"""
    global velocity_x, velocity_z, target_yaw, target_pitch
    global prev_left_knee_y, prev_right_knee_y, is_walking
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Get all key body points
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # === WALKING DETECTION (Knee Movement) ===
        if prev_left_knee_y is not None and prev_right_knee_y is not None:
            left_knee_movement = abs(left_knee.y - prev_left_knee_y)
            right_knee_movement = abs(right_knee.y - prev_right_knee_y)
            
            # Walking detected if knees are moving significantly
            is_walking = (left_knee_movement > step_threshold or right_knee_movement > step_threshold)
        
        prev_left_knee_y = left_knee.y
        prev_right_knee_y = right_knee.y
        
        # === BODY LEAN FOR DIRECTION ===
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_x = (left_hip.x + right_hip.x) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        
        # Forward/backward lean
        body_lean_forward = hip_center_y - shoulder_center_y
        # Left/right lean
        body_lean_side = hip_center_x - shoulder_center_x
        
        # === MOVEMENT CONTROL ===
        if is_walking:
            # Forward/backward based on body lean
            if body_lean_forward > 0.03:  # Leaning forward
                velocity_z -= acceleration
            elif body_lean_forward < -0.03:  # Leaning backward
                velocity_z += acceleration
            
            # Strafe left/right based on side lean
            if body_lean_side > 0.02:  # Leaning right
                velocity_x += acceleration
            elif body_lean_side < -0.02:  # Leaning left
                velocity_x -= acceleration
        
        # === HEAD ROTATION FOR CAMERA ===
        # Horizontal look (left/right)
        head_horizontal = (nose.x - 0.5) * 2  # -1 to 1
        target_yaw = -head_horizontal * 90  # -90 to +90 degrees
        
        # Vertical look (up/down) using nose position relative to shoulders
        nose_shoulder_diff = nose.y - shoulder_center_y
        target_pitch = nose_shoulder_diff * 60  # Look up/down
        target_pitch = max(-45, min(45, target_pitch))  # Limit pitch
        
        # === ARM GESTURES FOR SPEED BOOST ===
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # Arms raised = speed boost
        if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
            speed_multiplier = 1.5
            cv2.putText(frame, "SPEED BOOST!", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            speed_multiplier = 1.0
        
        # Apply speed multiplier
        velocity_x *= speed_multiplier
        velocity_z *= speed_multiplier
        
        # === JUMPING (Both arms raised high) ===
        if left_wrist.y < nose.y - 0.1 and right_wrist.y < nose.y - 0.1:
            cv2.putText(frame, "JUMP!", (10, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        # === DISPLAY HUD ===
        status = "WALKING" if is_walking else "STANDING"
        color = (0, 255, 0) if is_walking else (0, 165, 255)
        
        cv2.putText(frame, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, "WALK IN PLACE: Move Forward", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "LEAN BODY: Change Direction", (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "TURN HEAD: Look Around", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "RAISE ARMS: Speed Boost", (10, 135), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'Q' to Quit", (10, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return frame

def draw_ground():
    """Draw realistic frozen ground with texture"""
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # Main ice ground
    glColor4f(0.85, 0.95, 1.0, 1.0)
    glBegin(GL_QUADS)
    glVertex3f(-300, 0, -300)
    glVertex3f(300, 0, -300)
    glVertex3f(300, 0, 300)
    glVertex3f(-300, 0, 300)
    glEnd()
    
    # Add some ice cracks pattern
    glColor4f(0.7, 0.85, 0.95, 0.5)
    for i in range(20):
        x = random.uniform(-300, 300)
        z = random.uniform(-300, 300)
        glBegin(GL_LINES)
        glVertex3f(x, 0.01, z)
        glVertex3f(x + random.uniform(-10, 10), 0.01, z + random.uniform(-10, 10))
        glEnd()
    
    glDisable(GL_BLEND)

def draw_ice_crystal(crystal):
    """Draw detailed ice crystal with glow effect"""
    glPushMatrix()
    
    # Floating animation
    float_offset = math.sin(glutGet(GLUT_ELAPSED_TIME) * 0.001 * crystal.float_speed + crystal.offset) * 0.3
    glTranslatef(crystal.x, crystal.y + float_offset, crystal.z)
    glRotatef(crystal.rotation + glutGet(GLUT_ELAPSED_TIME) * 0.01 * crystal.rotation_speed, 0, 1, 0)
    
    # Semi-transparent crystal with glow
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # Outer glow
    glColor4f(0.6, 0.9, 1.0, 0.3)
    glutSolidCone(crystal.size * 1.3, crystal.height * 1.2, 8, 1)
    
    # Main crystal
    glColor4f(0.7, 0.95, 1.0, 0.8)
    glutSolidCone(crystal.size, crystal.height, 6, 1)
    
    # Top sparkle
    glTranslatef(0, crystal.height, 0)
    glColor4f(1.0, 1.0, 1.0, 0.9)
    glutSolidSphere(crystal.size * 0.2, 8, 8)
    
    glDisable(GL_BLEND)
    glPopMatrix()

def draw_frozen_tree(tree):
    """Draw detailed frozen tree"""
    glPushMatrix()
    glTranslatef(tree.x, tree.y, tree.z)
    
    # Trunk with ice coating
    glColor3f(0.65, 0.75, 0.85)
    glPushMatrix()
    glRotatef(-90, 1, 0, 0)
    gluCylinder(gluNewQuadric(), tree.trunk_radius, tree.trunk_radius * 0.8, tree.height * 0.6, 12, 1)
    glPopMatrix()
    
    # Multiple layers of frozen foliage
    num_layers = 4
    for i in range(num_layers):
        layer_height = tree.height * 0.6 + (i * tree.height * 0.15)
        layer_size = tree.foliage_radius * (1 - i * 0.2)
        
        glPushMatrix()
        glTranslatef(0, layer_height, 0)
        glColor4f(0.75 + i * 0.05, 0.88, 0.95, 0.9)
        glRotatef(-90, 1, 0, 0)
        glutSolidCone(layer_size, tree.height * 0.25, 10, 1)
        glPopMatrix()
    
    # Snow on top
    glTranslatef(0, tree.height, 0)
    glColor3f(1.0, 1.0, 1.0)
    glutSolidSphere(tree.trunk_radius * 0.8, 8, 8)
    
    glPopMatrix()

def draw_ice_boulder(boulder):
    """Draw ice boulder"""
    glPushMatrix()
    glTranslatef(boulder.x, boulder.y, boulder.z)
    glColor4f(0.8, 0.92, 0.98, 0.85)
    
    # Irregular shape using multiple scaled spheres
    glutSolidSphere(boulder.size, 12, 12)
    glScalef(1.2, 0.8, 1.1)
    glutSolidSphere(boulder.size * 0.8, 10, 10)
    
    glPopMatrix()

def draw_snowflake(flake):
    """Draw individual snowflake"""
    glPushMatrix()
    glTranslatef(flake.x, flake.y, flake.z)
    glColor4f(1.0, 1.0, 1.0, 0.9)
    glutSolidSphere(flake.size, 4, 4)
    glPopMatrix()

def draw_sky():
    """Draw atmospheric sky with gradient"""
    glDisable(GL_LIGHTING)
    glBegin(GL_QUADS)
    
    # Top (lighter)
    glColor3f(0.75, 0.88, 0.95)
    glVertex3f(-300, 50, -300)
    glVertex3f(300, 50, -300)
    glVertex3f(300, 50, 300)
    glVertex3f(-300, 50, 300)
    
    glEnd()
    glEnable(GL_LIGHTING)

def display():
    """Main display with realistic rendering"""
    global player_x, player_y, player_z, velocity_x, velocity_z
    global camera_yaw, camera_pitch, target_yaw, target_pitch
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    # Smooth camera rotation
    camera_yaw += (target_yaw - camera_yaw) * 0.1
    camera_pitch += (target_pitch - camera_pitch) * 0.1
    
    # Apply friction to velocity
    velocity_x *= friction
    velocity_z *= friction
    
    # Clamp velocity to max speed
    speed = math.sqrt(velocity_x**2 + velocity_z**2)
    if speed > max_speed:
        velocity_x = (velocity_x / speed) * max_speed
        velocity_z = (velocity_z / speed) * max_speed
    
    # Update player position with rotation
    rad = math.radians(camera_yaw)
    player_x += velocity_x * math.cos(rad) - velocity_z * math.sin(rad)
    player_z += velocity_x * math.sin(rad) + velocity_z * math.cos(rad)
    
    # Camera setup with pitch and yaw
    look_x = player_x + math.sin(rad)
    look_y = player_y + math.tan(math.radians(camera_pitch))
    look_z = player_z + math.cos(rad)
    
    gluLookAt(
        player_x, player_y, player_z,
        look_x, look_y, look_z,
        0, 1, 0
    )
    
    # Draw world elements
    draw_sky()
    draw_ground()
    
    # Draw all objects
    for crystal in ice_crystals:
        # Only draw objects within view distance for performance
        dist = math.sqrt((crystal.x - player_x)**2 + (crystal.z - player_z)**2)
        if dist < 150:
            draw_ice_crystal(crystal)
    
    for tree in frozen_trees:
        dist = math.sqrt((tree.x - player_x)**2 + (tree.z - player_z)**2)
        if dist < 150:
            draw_frozen_tree(tree)
    
    for boulder in ice_boulders:
        dist = math.sqrt((boulder.x - player_x)**2 + (boulder.z - player_z)**2)
        if dist < 150:
            draw_ice_boulder(boulder)
    
    # Update and draw snowflakes
    for flake in snowflakes:
        draw_snowflake(flake)
        
        # Update snowflake physics
        flake.y -= flake.speed
        flake.x += flake.drift_x
        flake.z += flake.drift_z
        
        # Reset snowflake when it hits ground
        if flake.y < 0:
            flake.y = 80
            flake.x = player_x + random.uniform(-100, 100)
            flake.z = player_z + random.uniform(-100, 100)
    
    glutSwapBuffers()

def timer(value):
    """Animation timer"""
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frame = process_pose(frame)
        cv2.imshow('Body Tracking - Frozen World Explorer', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()
    
    glutPostRedisplay()
    glutTimerFunc(16, timer, 0)

def init_gl():
    """Initialize OpenGL with enhanced settings"""
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glShadeModel(GL_SMOOTH)
    
    # Enhanced lighting
    glLightfv(GL_LIGHT0, GL_POSITION, [50, 100, 50, 1])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.7, 0.75, 0.8, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.9, 0.95, 1.0, 1])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1])
    
    # Background
    glClearColor(0.68, 0.85, 0.92, 1.0)
    
    # Enhanced fog
    glEnable(GL_FOG)
    glFogi(GL_FOG_MODE, GL_LINEAR)
    glFogfv(GL_FOG_COLOR, [0.75, 0.88, 0.95, 1.0])
    glFogf(GL_FOG_START, 50.0)
    glFogf(GL_FOG_END, 150.0)
    glFogf(GL_FOG_DENSITY, 0.02)

def reshape(width, height):
    """Handle window reshape"""
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(80, width / height, 0.1, 500.0)
    glMatrixMode(GL_MODELVIEW)

def main():
    """Main function"""
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(1280, 720)
    glutCreateWindow(b"Frozen World - Realistic Body Tracking")
    
    init_gl()
    
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutTimerFunc(0, timer, 0)
    
    print("=" * 70)
    print("       FROZEN WORLD - REALISTIC MOVEMENT CONTROL")
    print("=" * 70)
    print(" WALK IN PLACE (move knees): Move forward/backward")
    print(" LEAN YOUR BODY: Control direction while walking")
    print(" TURN YOUR HEAD LEFT/RIGHT: Look around horizontally")
    print(" LOOK UP/DOWN: Change camera pitch")
    print(" RAISE BOTH ARMS: Speed boost!")
    print(" RAISE ARMS HIGH: Jump!")
    print()
    print(" Press 'Q' in camera window to quit")
    print("=" * 70)
    
    glutMainLoop()

if __name__ == "__main__":
    main()