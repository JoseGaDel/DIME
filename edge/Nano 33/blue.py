'''
This code is designed to run on the central device to measure BLE communication times with the board.
It uses the simplepyble library to connect to the peripheral device and read the data from the characteristic.
Once the image is received entirely, it sends a byte back representing the class predicted by the LML model.
'''

import simplepyble
import numpy as np
import time

lista = []   
num = 0

if __name__ == "__main__":
    adapters = simplepyble.Adapter.get_adapters()

    if len(adapters) == 0:
        print("No adapters found")

    # Query the user to pick an adapter
    print("Please select an adapter:")
    for i, adapter in enumerate(adapters):
        print(f"{i}: {adapter.identifier()} [{adapter.address()}]")

    choice = int(input("Enter choice: "))
    adapter = adapters[choice]

    print(f"Selected adapter: {adapter.identifier()} [{adapter.address()}]")

    adapter.set_callback_on_scan_start(lambda: print("Scan started."))
    adapter.set_callback_on_scan_stop(lambda: print("Scan complete."))
    adapter.set_callback_on_scan_found(lambda peripheral: print(f"Found {peripheral.identifier()} [{peripheral.address()}]"))

    # Scan for 5 seconds
    adapter.scan_for(5000)
    peripherals = adapter.scan_get_results()

    # Query the user to pick a peripheral
    print("Please select a peripheral:")
    for i, peripheral in enumerate(peripherals):
        print(f"{i}: {peripheral.identifier()} [{peripheral.address()}]")

    choice = int(input("Enter choice: "))
    peripheral = peripherals[choice]

    service_uuid = '19b10000-e8f2-537e-4f6c-d104768a1214'
    characteristic_uuid= ['19b10001-e8f2-537e-4f6c-d104768a1214',
                          '19b10002-e8f2-537e-4f6c-d104768a1214',
                          '19b10003-e8f2-537e-4f6c-d104768a1214',
                          '19b10004-e8f2-537e-4f6c-d104768a1214',
                          '19b10005-e8f2-537e-4f6c-d104768a1214',
                          '19b10006-e8f2-537e-4f6c-d104768a1214',
                          '19b10007-e8f2-537e-4f6c-d104768a1214',
                          '19b10008-e8f2-537e-4f6c-d104768a1214',
                          '19b10009-e8f2-537e-4f6c-d104768a1214',
                          '19b10010-e8f2-537e-4f6c-d104768a1214',
                          '19b10011-e8f2-537e-4f6c-d104768a1214',
                          '19b10012-e8f2-537e-4f6c-d104768a1214',
                          '19b10013-e8f2-537e-4f6c-d104768a1214',
                          '19b10014-e8f2-537e-4f6c-d104768a1214',]
    

    def callback(data):
        global lista
        lista.extend(data)    
        
    # Define callback function on connection
    def callback_con():
        for i in range(13): 
            # The function below subscribes to the characteristics, it rensponds with "callback" 
            peripheral.notify(service_uuid, characteristic_uuid[i],  callback)
      

    print(f"Connecting to: {peripheral.identifier()} [{peripheral.address()}]")


    peripheral.set_callback_on_connected(callback_con)
    peripheral.connect() 

    while len(lista) != 3072:
        time.sleep(0.001)
   
    # Send byte back once the whole image has been received
    message = 5
    peripheral.write_command(service_uuid,characteristic_uuid[-1], message.to_bytes(1,'little')) 
    
    print(lista)
    print(len(lista))  


    peripheral.disconnect()
    print("disconnected") 
    
   
    
  
