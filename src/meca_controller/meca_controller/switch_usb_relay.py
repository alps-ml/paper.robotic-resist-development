import serial
import time


usb_relay = serial.Serial("/dev/ttyUSB0",9600)
if usb_relay.is_open:
   print(usb_relay)
   on_cmd = b'\xA0\x01\x01\xa2'
   off_cmd =  b'\xA0\x01\x00\xa1'


   usb_relay.write(on_cmd )
   time.sleep(5)
   usb_relay.write(off_cmd)
   time.sleep(1)
   
usb_relay.close()

# if disconnecting and reconnecting the usb relay, run the following command
# sudo chmod 666 /dev/ttyUSB0