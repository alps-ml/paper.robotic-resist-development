# ADJUST THESE GLOBAL CONSTANTS:
ROBOT1 = {'namespace':'/robot1', 
          'ip':'192.168.0.100',
          'command_buffer_time': 0.25, # seconds, For commands that are sent upfront, how big should the time buffer be with respect to execution time.
          'monitoring_interval': 0.05, # Set monitoring interval for real time data refresh time
          'move_speed': 40             # Speed for moving between positions
         }
