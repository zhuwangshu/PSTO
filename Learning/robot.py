import numpy as np
import serial
import time

from utils import *


class AntRobot():
  def __init__(self, port='COM3'):
    print('COM3')
  # def __init__(self, port='/dev/tty.BT04-A-Port'):
    
    self.port = port
    self.ser = serial.Serial(self.port,9600,timeout=.1)
    self.angles = np.zeros(8)
    # self.dead_height = 0.4
    time.sleep(1)

  def reset(self):
    '''
    Set the robot to initial posture
    '''
    # self.sendSignals([300]*8)
    for _ in range(20):
      self.sendSignals([330]*8)

    obs = self.read()
    self.angles = obs[0:8]
    print("obs",obs)
    return obs

  def step(self, angles):
    '''
    Send actions and return observations
    '''
    self.send(angles)
    # print("After sending ...")
    obs = self.read()
    self.angles = obs[0:8]
    # print("After receiving ...")
    # done = obs[-1] > self.dead_height
    done = 0
    return obs, done


  #######################################################
  # Functions for communication between computer and real robot

  def send(self, angles):
    '''
    Control each servo motor by angles
    '''

    # signals = angle2signal(angles)
    signals = angle2signal(angles+ (np.random.rand(8)-0.5) * 1) # add noise
    signals =  self.angles - signals  
    signals = np.clip(signals,290,370)
    # signals[0:4] = np.clip(signals[0:4],310,350)
    # signals[4:8] = np.clip(signals[4:8],230,290)
    self.sendSignals(signals)

  def sendSignals(self, signals):
    '''
    Control each servo motor by 
    '''
    def encode(signals):
      '''
      Encode signals into data package
      '''
      message = '^'
      for i, signal in enumerate(signals):
        message += str(i+1) + '=' + str(signal) + '&'
      message += '$'
      message = bytes(message, encoding='UTF-8')
      return message
    message = encode(signals)
    # print("message",message)
    # while 1:
    self.ser.write(message)
    # print("message sent",message)

  def read(self):
    '''
    Get IMU readings
    '''
    # self.ser.readline()
    # message = self.ser.readline()
    # print("message read",message)

    # for _ in range(10):
    message = self.ser.readline()
    # print("messagebuff",messagebuff)
    # if (messagebuff!=b''):
    #   message = messagebuff
    #   print("!!!!!!!!messagebuff !=b''",messagebuff) 
    # while (message==b''):
    #   message = self.ser.readline()
    #strip out the new lines for now
    # print("message read",message)
    message = str(message, 'utf-8')
    message = message.strip('^$')
    states = message.split('&')
    print(states[-2])  # power per step
    # print("average power ",states[-1], " power ", states[-2] ) # average power
    states = [float(s) for s in states]
    states[-1] = states[-1] /10 
      
    return states
  
  def connect(self):
    if not self.ser.is_open:
      self.ser = serial.Serial(self.port)
      time.sleep(1)
    return self.ser.is_open

  def disconnect(self):
    if self.ser.is_open:
      time.sleep(1)
      self.ser.close()
    return self.ser.is_open

  def __del__(self):
    if self.ser.is_open:
      time.sleep(1)
      self.ser.close()

  #######################################################


