#!/usr/bin/python3
#
# Connects to a specified device, starts data characteristic notifications and logs values
# to the console as they are received in PropertiesChanged signals.
#
# Run from the command line with a bluetooth device address argument

import bluetooth_constants
import bluetooth_utils
import dbus
import dbus.mainloop.glib
import sys
import time
from gi.repository import GLib
sys.path.insert(0, '.')

bus = None
device_interface = None
device_path = None
found_offload_service = False
found_offload_characteristic  = False
offload_service_path = None
offload_characteristic_path  = None
found_response_service = False
found_response_characteristic  = False
response_service_path = None
response_characteristic_path  = None
text = None
image_size = 3072
received_size = 0
it = 0

def write_text(text):
    global response_characteristic_path
    char_proxy = bus.get_object(bluetooth_constants.BLUEZ_SERVICE_NAME, response_characteristic_path)
    char_interface = dbus.Interface(char_proxy, bluetooth_constants.GATT_CHARACTERISTIC_INTERFACE)
    try:
        ascii = bluetooth_utils.text_to_ascii_array(text)
        value = char_interface.WriteValue(ascii, {})
    except Exception as e:
        print("Failed to write to Central device")
        print(e.get_dbus_name())
        print(e.get_dbus_message())
        return bluetooth_constants.RESULT_EXCEPTION
    else:
        return bluetooth_constants.RESULT_OK


def data_received(interface, changed, invalidated, path):
    global received_size
    global it
    if 'Value' in changed:
        data = bluetooth_utils.dbus_to_python(changed['Value'])
        received_size += len(data)
        if received_size >= image_size:
            received_size = 0
            it += 1
            write_text(text)


def start_notifications():
    global offload_characteristic_path
    global bus
    char_proxy = bus.get_object(bluetooth_constants.BLUEZ_SERVICE_NAME, offload_characteristic_path)
    char_interface = dbus.Interface(char_proxy, bluetooth_constants.GATT_CHARACTERISTIC_INTERFACE)

    bus.add_signal_receiver(data_received,
        dbus_interface = bluetooth_constants.DBUS_PROPERTIES,
        signal_name = "PropertiesChanged",
        path = offload_characteristic_path,
        path_keyword = "path")
    
    try:
        print("Starting notifications")
        char_interface.StartNotify()
        print("Done starting notifications")
    except Exception as e:
        print("Failed to start data notifications")
        print(e.get_dbus_name())
        print(e.get_dbus_message())
        return bluetooth_constants.RESULT_EXCEPTION
    else:
        return bluetooth_constants.RESULT_OK
    
def service_discovery_completed():
    global found_offload_service
    global found_offload_characteristic
    global offload_service_path
    global offload_characteristic_path
    global bus
    global found_response_service
    global found_response_characteristic
    global response_service_path
    global response_characteristic_path
    global text
    
    if found_offload_service and found_offload_characteristic:
        print("Required service and characteristic found - device is OK")
        print("data service path: ",offload_service_path)
        print("data characteristic path: ",offload_characteristic_path)
        start_notifications()
    else:
        print("Required service and characteristic were not found - device is NOK")
        print("data service found: ",str(found_offload_service))
        print("data characteristic found: ",str(found_offload_characteristic))
    if found_response_service and found_response_characteristic:
        print("Required service and characteristic found - device is OK")
        print("Response service path: ",response_service_path)
        print("Response characteristic path: ",response_characteristic_path)
    else:
        print("Required service and characteristic were not found - device is NOK")
        print("Response service found: ",str(found_response_service))
        print("Response Text characteristic found: ",str(found_response_characteristic))
    bus.remove_signal_receiver(interfaces_added,"InterfacesAdded")
    bus.remove_signal_receiver(properties_changed,"PropertiesChanged")
    
def properties_changed(interface, changed, invalidated, path):
    global device_path
    if path != device_path:
        return

    if 'ServicesResolved' in changed:
        sr = bluetooth_utils.dbus_to_python(changed['ServicesResolved'])
        print("ServicesResolved  : ", sr)
        if sr == True:
            service_discovery_completed()
        

def interfaces_added(path, interfaces):
    global found_offload_service
    global found_offload_characteristic
    global offload_service_path
    global offload_characteristic_path
    global found_response_service
    global found_response_characteristic
    global response_service_path
    global response_characteristic_path
    if bluetooth_constants.GATT_SERVICE_INTERFACE in interfaces:
        properties = interfaces[bluetooth_constants.GATT_SERVICE_INTERFACE]
        print("--------------------------------------------------------------------------------")
        print("SVC path   :", path)
        if 'UUID' in properties:
            uuid = properties['UUID']
            if uuid == bluetooth_constants.OFFLOAD_SVC_UUID:
                found_offload_service = True
                offload_service_path = path
            if uuid == bluetooth_constants.RESPONSE_SVC_UUID:
                found_response_service = True
                response_service_path = path
            print("SVC UUID   : ", bluetooth_utils.dbus_to_python(uuid))
            print("SVC name   : ", bluetooth_utils.get_name_from_uuid(uuid))
        return

    if bluetooth_constants.GATT_CHARACTERISTIC_INTERFACE in interfaces:
        properties = interfaces[bluetooth_constants.GATT_CHARACTERISTIC_INTERFACE]
        print("  CHR path   :", path)
        if 'UUID' in properties:
            uuid = properties['UUID']
            if uuid == bluetooth_constants.OFFLOAD_CHR_UUID:
                found_offload_characteristic = True
                offload_characteristic_path = path
            if uuid == bluetooth_constants.RESPONSE_CHR_UUID:
                found_response_characteristic = True
                response_characteristic_path = path
            print("  CHR UUID   : ", bluetooth_utils.dbus_to_python(uuid))
            print("  CHR name   : ", bluetooth_utils.get_name_from_uuid(uuid))
            flags  = ""
            for flag in properties['Flags']:
                flags = flags + flag + ","
            print("  CHR flags  : ", flags)
        return
    
    if bluetooth_constants.GATT_DESCRIPTOR_INTERFACE in interfaces:
        properties = interfaces[bluetooth_constants.GATT_DESCRIPTOR_INTERFACE]
        print("    DSC path   :", path)
        if 'UUID' in properties:
            uuid = properties['UUID']
            print("    DSC UUID   : ", bluetooth_utils.dbus_to_python(uuid))
            print("    DSC name   : ", bluetooth_utils.get_name_from_uuid(uuid))
        return

def connect():
    global bus
    global device_interface
    try:
        device_interface.Connect()
    except Exception as e:
        print("Failed to connect")
        print(e.get_dbus_name())
        print(e.get_dbus_message())
        if ("UnknownObject" in e.get_dbus_name()):
            print("Try scanning first to resolve this problem")
        return bluetooth_constants.RESULT_EXCEPTION
    else:
        print("Connected OK")
        return bluetooth_constants.RESULT_OK

if (len(sys.argv) != 2):
    print("usage: python3 client_monitor_data.py [bdaddr]")
    sys.exit(1)

bdaddr = sys.argv[1]
text='111'
dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
bus = dbus.SystemBus()
adapter_path = bluetooth_constants.BLUEZ_NAMESPACE + bluetooth_constants.ADAPTER_NAME
device_path = bluetooth_utils.device_address_to_path(bdaddr, adapter_path)
device_proxy = bus.get_object(bluetooth_constants.BLUEZ_SERVICE_NAME,device_path)
device_interface = dbus.Interface(device_proxy, bluetooth_constants.DEVICE_INTERFACE)

print("Connecting to " + bdaddr)
connect()
print("Discovering services++")
print("Registering to receive InterfacesAdded signals")
bus.add_signal_receiver(interfaces_added,
        dbus_interface = bluetooth_constants.DBUS_OM_IFACE,
        signal_name = "InterfacesAdded")
print("Registering to receive PropertiesChanged signals")
bus.add_signal_receiver(properties_changed,
        dbus_interface = bluetooth_constants.DBUS_PROPERTIES,
        signal_name = "PropertiesChanged",
        path_keyword = "path")
mainloop = GLib.MainLoop()

mainloop.run()
