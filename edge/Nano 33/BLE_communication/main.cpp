/*
This code has been designed to run on the Arduino Nano 33 BLE Sense. The IDE used is VSCode with the PlatformIO extension.
The objective is to measure the BLE communication time between the board and a pc. A 3072B image is sent and a byte back 
representing the predicted class is sent back to the board.
*/

#include <Arduino.h>
#include <ArduinoBLE.h>
#include "image.h"


const int characteristicSize = 244; 
byte value = 0;
int counter = 0;
uint8_t receivedByte;

int start;
int end;
int middle;

// Create a service
BLEService customService("19B10000-E8F2-537E-4F6C-D104768A1214"); 

// Create characteristics
BLECharacteristic customCharacteristics[13] = {
  BLECharacteristic("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite | BLEWriteWithoutResponse | BLENotify | BLEIndicate, characteristicSize),
  BLECharacteristic("19B10002-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite | BLEWriteWithoutResponse | BLENotify | BLEIndicate, characteristicSize),
  BLECharacteristic("19B10003-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite | BLEWriteWithoutResponse | BLENotify | BLEIndicate, characteristicSize),
  BLECharacteristic("19B10004-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite | BLEWriteWithoutResponse | BLENotify | BLEIndicate, characteristicSize),
  BLECharacteristic("19B10005-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite | BLEWriteWithoutResponse | BLENotify | BLEIndicate, characteristicSize),
  BLECharacteristic("19B10006-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite | BLEWriteWithoutResponse | BLENotify | BLEIndicate, characteristicSize),
  BLECharacteristic("19B10007-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite | BLEWriteWithoutResponse | BLENotify | BLEIndicate, characteristicSize),
  BLECharacteristic("19B10008-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite | BLEWriteWithoutResponse | BLENotify | BLEIndicate, characteristicSize),
  BLECharacteristic("19B10009-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite | BLEWriteWithoutResponse | BLENotify | BLEIndicate, characteristicSize),
  BLECharacteristic("19B10010-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite | BLEWriteWithoutResponse | BLENotify | BLEIndicate, characteristicSize),
  BLECharacteristic("19B10011-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite | BLEWriteWithoutResponse | BLENotify | BLEIndicate, characteristicSize),
  BLECharacteristic("19B10012-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite | BLEWriteWithoutResponse | BLENotify | BLEIndicate, characteristicSize), 
  BLECharacteristic("19B10013-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite | BLEWriteWithoutResponse | BLENotify | BLEIndicate, 144)
};
BLEByteCharacteristic classCharacteristic("19B10014-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite | BLEWriteWithoutResponse | BLENotify);


// Define function for the event handler. Once the byte back is received it stops the time and disconnects. 
void switchCharacteristicWritten(BLEDevice central, BLECharacteristic characteristic) {
  end = micros();
  Serial.println("time: "+ String(end-start));
  int value = classCharacteristic.value();
  Serial.println("value: "+ String(value));
  BLE.disconnect();
  Serial.println("Disconnected");
}


void setup() {
  Serial.begin(115200);
  while (!Serial){};
  
  // Start BLE
  if (!BLE.begin()) {
    Serial.println("Starting BLE failed!");
    while (1);
  }

  // Set the chosen connection interval
  BLE.setConnectionInterval(0x0006, 0x0006);


  // Set the local name for the BLE device
  BLE.setLocalName("Nano33BLE");


  // Add the service and characteristics
  BLE.setAdvertisedService(customService);
  for (int i = 0; i < 13; i++) {
    customService.addCharacteristic(customCharacteristics[i]);
  }
  customService.addCharacteristic(classCharacteristic);
  BLE.addService(customService);

  // Set the previously defined event handler
  classCharacteristic.setEventHandler(BLEWritten, switchCharacteristicWritten);

  // Start advertising
  BLE.advertise();

}

void loop() {

  BLEDevice central = BLE.central();

  while (central.connected()) {
   
    // Once there is connection and subscription to the characterisitcs from the central side, the image is sent
    if (customCharacteristics[12].subscribed() && counter ==0){ 
      start = micros();
      for (int i = 0; i < 12; i++) {
        customCharacteristics[i].writeValue(&test_image_0[i*244], characteristicSize, false);
        // Without this delay, the image is not received completely by the central device.
        delay(9);
      }   
      customCharacteristics[12].writeValue(&test_image_0[2928], 144, false);
      counter++;
    }      
      
      
  }    
  
}
