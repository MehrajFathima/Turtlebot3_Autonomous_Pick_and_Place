#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        
        # Node 1: Detector Node (YOLOv8 object detection)
        Node(
            package='tb3_autonomous_pick_place',
            executable='detector_node',
            name='detector_node',
            output='screen'
        ),

        # Node 2: Fusion Node (sensor fusion/data processing)
        Node(
            package='tb3_autonomous_pick_place',
            executable='fusion_node',
            name='fusion_node',
            output='screen'
        ),

        # Node 3: Home Node (homing/initial positioning)
        Node(
            package='tb3_autonomous_pick_place',
            executable='ik_node',
            name='ik_node',
            output='screen'
        ),

        # Node 4: Move Node (arm movement/IK control)
        Node(
            package='tb3_autonomous_pick_place',
            executable='move_node',
            name='move_node',
            output='screen'
        ),
    ])