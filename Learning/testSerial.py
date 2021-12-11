import serial, time

arduino = serial.Serial('COM3', 9600, timeout=.1)

time.sleep(1) #give the connection a second to settle

arduino.write(b'^1=300&2=300&3=300&4=300&5=300&6=300&7=300&8=300&$')

while True:

	data = arduino.readline()

	if data:
	    print(data) #strip out the new lines for now
	arduino.write(b'^1=300&2=300&3=300&4=300&5=300&6=300&7=300&8=300&$')

# （better to do .read() in the long run for this reason