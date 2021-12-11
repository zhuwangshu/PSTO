#include <Adafruit_PWMServoDriver.h>
#include <SoftwareSerial.h>
#include <Wire.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
SoftwareSerial BT(8,7); // RX-8, TX-7

const int MPU6050_addr=0x68;
int16_t AccX,AccY,AccZ,Temp,GyroX,GyroY,GyroZ;
unsigned int Height = 0;

void setup() {
  Serial.begin(9600); // USB debug serial
  BT.begin(9600); // Bluetooth serial
  pwm.begin();pwm.setPWMFreq(60);
  delay(100);
  for (int i=8;i>0;i--) {
    pwm.setPWM(i, 0, 300);
    delay(100);
  }
  Wire.begin();Wire.beginTransmission(MPU6050_addr);
  Wire.write(0x6B);Wire.write(0);
  Wire.endTransmission(true);
  Serial.println("All ready!");
}

void loop() {
  if (BT.available() > 0) {
    String comdata = BT.readStringUntil('$');
    if (comdata.startsWith(String('^'))) {
      int index = 0, angles[8] = {0,0,0,0,0,0,0,0};
      comdata = comdata.substring(1, comdata.length());
      index = comdata.indexOf('&');
      // parse message into servo id and angle
      while (index != -1) {
        String op = comdata.substring(0, index);
        int id = op.substring(0, op.indexOf('=')).toInt();
        int angle = op.substring(op.indexOf('=')+1).toInt();
        comdata = comdata.substring(index+1);
        if (id != 0 && angle != 0) {
          angles[id-1] = angle;
        }
        index = comdata.indexOf('&');
      }
      // Controlling servo motors
      Serial.println("Controlling servo motors");
      for (int i=0;i<8;i++) {
        if (angles[i] != 0) {
          pwm.setPWM(i+1, 0, angles[i]);
          Serial.println("Joint " + String(i+1) + ' ' + " rotate " + String(angles[i]) + ' ' + " degree");
        }
      }
      sendData();
    }
  }
}


void sendData() {
  Wire.beginTransmission(MPU6050_addr);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_addr,14,true);
  AccX=Wire.read()<<8|Wire.read();AccY=Wire.read()<<8|Wire.read();AccZ=Wire.read()<<8|Wire.read();
  Temp=Wire.read()<<8|Wire.read();
  GyroX=Wire.read()<<8|Wire.read();GyroY=Wire.read()<<8|Wire.read();GyroZ=Wire.read()<<8|Wire.read();
  BT.print("^");
  BT.print(AccX/32768.0*2); BT.print("&");
  BT.print(AccY/32768.0*2); BT.print("&");
  BT.print(AccZ/32768.0*2); BT.print("&");
//BT.print(Temp/340.00+36.53); BT.print("&");
  BT.print(GyroX/32768.0*250); BT.print("&");
  BT.print(GyroY/32768.0*250); BT.print("&");
  BT.print(GyroZ/32768.0*250); BT.print("&");
  delay(500);
  Height = analogRead(0);
  BT.print(Height);BT.print("\n");
}
