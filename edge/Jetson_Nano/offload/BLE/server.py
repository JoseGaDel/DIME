#!/usr/bin/python3
# Advertises, accepts a connection, creates the offloading service and characteristic

import bluetooth_constants
import bluetooth_gatt
import bluetooth_exceptions
import bluetooth_utils
import dbus
import dbus.exceptions
import dbus.service
import dbus.mainloop.glib
import sys
import random
from gi.repository import GObject
from gi.repository import GLib
sys.path.insert(0, '.')

DATA_PATH='../inference/data/test_batch.bin'

bus = None
adapter_path = None
adv_mgr_interface = None
connected = 0
awaiting_response = False
start_time = 0.0
end_time = 0.0
current_instance = 0
max_instances = 30
latencies = [0.0] * max_instances
image_size = 3072
packet_size = 244

# much of this code was copied or inspired by test\example-advertisement in the BlueZ source
class Advertisement(dbus.service.Object):
    PATH_BASE = '/org/bluez/ldsg/advertisement'

    def __init__(self, bus, index, advertising_type):
        self.path = self.PATH_BASE + str(index)
        self.bus = bus
        self.ad_type = advertising_type
        self.service_uuids = None
        self.manufacturer_data = None
        self.solicit_uuids = None
        self.service_data = None
        self.local_name = 'HI Central'
        self.include_tx_power = False
        self.data = None
        self.discoverable = True
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        properties = dict()
        properties['Type'] = self.ad_type
        if self.service_uuids is not None:
            properties['ServiceUUIDs'] = dbus.Array(self.service_uuids,
                                                    signature='s')
        if self.solicit_uuids is not None:
            properties['SolicitUUIDs'] = dbus.Array(self.solicit_uuids,
                                                    signature='s')
        if self.manufacturer_data is not None:
            properties['ManufacturerData'] = dbus.Dictionary(
                self.manufacturer_data, signature='qv')
        if self.service_data is not None:
            properties['ServiceData'] = dbus.Dictionary(self.service_data,
                                                        signature='sv')
        if self.local_name is not None:
            properties['LocalName'] = dbus.String(self.local_name)
        if self.discoverable is not None and self.discoverable == True:
            properties['Discoverable'] = dbus.Boolean(self.discoverable)
        if self.include_tx_power:
            properties['Includes'] = dbus.Array(["tx-power"], signature='s')

        if self.data is not None:
            properties['Data'] = dbus.Dictionary(
                self.data, signature='yv')
        print(properties)
        return {bluetooth_constants.ADVERTISING_MANAGER_INTERFACE: properties}

    def get_path(self):
        return dbus.ObjectPath(self.path)

    @dbus.service.method(bluetooth_constants.DBUS_PROPERTIES,
                         in_signature='s',
                         out_signature='a{sv}')
    def GetAll(self, interface):
        if interface != bluetooth_constants.ADVERTISEMENT_INTERFACE:
            raise bluetooth_exceptions.InvalidArgsException()
        return self.get_properties()[bluetooth_constants.ADVERTISING_MANAGER_INTERFACE]

    @dbus.service.method(bluetooth_constants.ADVERTISING_MANAGER_INTERFACE,
                         in_signature='',
                         out_signature='')
    def Release(self):
        print('%s: Released' % self.path)


class Application(dbus.service.Object):
    """
    org.bluez.GattApplication1 interface implementation
    """
    def __init__(self, bus):
        self.path = '/'
        self.services = []
        dbus.service.Object.__init__(self, bus, self.path)
        print("Adding OffloadService to the Application")
        self.add_service(OffloadService(bus, '/org/bluez/ldsg', 0))
        self.add_service(ResponseService(bus, '/org/bluez/ldsg', 1))

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_service(self, service):
        self.services.append(service)

    @dbus.service.method(bluetooth_constants.DBUS_OM_IFACE, out_signature='a{oa{sa{sv}}}')
    def GetManagedObjects(self):
        response = {}
        print('GetManagedObjects')

        for service in self.services:
            print("GetManagedObjects: service="+service.get_path())
            response[service.get_path()] = service.get_properties()
            chrcs = service.get_characteristics()
            for chrc in chrcs:
                response[chrc.get_path()] = chrc.get_properties()
                descs = chrc.get_descriptors()
                for desc in descs:
                    response[desc.get_path()] = desc.get_properties()

        return response

class OffloadService(bluetooth_gatt.Service):

     def __init__(self, bus, path_base, index):
        print("Initialising OffloadService object")
        bluetooth_gatt.Service.__init__(self, bus, path_base, index, bluetooth_constants.OFFLOAD_SVC_UUID, True)
        print("Adding OffloadCharacteristic to the service")
        self.add_characteristic(OffloadCharacteristic(bus, 0, self))

class OffloadCharacteristic(bluetooth_gatt.Characteristic):
    notifying = False
    
    def __init__(self, bus, index, service):
        bluetooth_gatt.Characteristic.__init__(
                self, bus, index,
                bluetooth_constants.OFFLOAD_CHR_UUID,
                ['read','notify'],
                service)
        self.notifying = False
        self.index = 0
        self.data = [dbus.Byte(i // packet_size) for i in range(image_size)] # Dummy data for testing. Elements are assigned a value depending on the packet they should belong, so is easy to check if the data is being sent correctly
        self.dataSize = len(self.data)

        GLib.timeout_add(0.5, self.offloading)

    def offloading(self):   
        # This would be where the offloading decision would take place
        if self.notifying:
            self.notify_image()

        GLib.timeout_add(1, self.offloading)

    def ReadValue(self, options):
        print('ReadValue in OffloadCharacteristic called')
        print('Returning '+str(self.data))
        value = []
        value.append(dbus.Byte(self.data))
        return value
    

    def notify_image(self):
        global awaiting_response
        global start_time
        if awaiting_response:
            return False
        if self.index == 0:
            start_time = GLib.get_monotonic_time()
        # notify the values of self.data
        # only `packet_size` elements will be sent, so we need to keep track of the index
        # and send the data in chunks
        start = self.index
        end = min(self.index + packet_size, self.dataSize)
        value = self.data[start:end]
        self.index += packet_size
        print("notifying data: ")
        self.PropertiesChanged(bluetooth_constants.GATT_CHARACTERISTIC_INTERFACE, { 'Value': value }, [])
        
        if self.index >= self.dataSize:
            self.index = 0
            # wait for the response message from the other end
            awaiting_response = True

        
        return self.notifying

    # note this overrides the same method in bluetooth_gatt.Characteristic where it is exported to 
    # make it visible over DBus
    def StartNotify(self):
        print("starting notifications")
        self.notifying = True

    def StopNotify(self):
        print("stopping notifications")
        self.notifying = False


class ResponseService(bluetooth_gatt.Service):
#   Service for receiving the result from the server

     def __init__(self, bus, path_base, index):
        print("Initialising ResponseService object")
        bluetooth_gatt.Service.__init__(self, bus, path_base, index, bluetooth_constants.RESPONSE_SVC_UUID, True)
        print("Adding ResponseCharacteristic to the service")
        self.add_characteristic(ResponseCharacteristic(bus, 0, self))

class ResponseCharacteristic(bluetooth_gatt.Characteristic):

    text = ""
    
    def __init__(self, bus, index, service):
        bluetooth_gatt.Characteristic.__init__(
                self, bus, index,
                bluetooth_constants.RESPONSE_CHR_UUID,
                ['write-without-response'],
                service)

    def WriteValue(self, value, options):
        global awaiting_response
        global start_time
        global end_time
        global current_instance
        global latencies

        end_time = GLib.get_monotonic_time()
        ascii_bytes = bluetooth_utils.dbus_to_python(value)
        latencies[current_instance] = (end_time - start_time) / 1000
        current_instance += 1

        if current_instance == max_instances:
            # save latencies to file and terminate
            with open('latencies.txt', 'w') as f:
                for latency in latencies:
                    f.write(str(latency) + '\n')
            print('latencies written to file: latencies.txt')
            mainloop.quit()
            
        text = ''.join(chr(i) for i in ascii_bytes)
        print(str(ascii_bytes) + " = " + text)
        awaiting_response = False



def register_ad_cb():
    print('Advertisement registered OK')

def register_ad_error_cb(error):
    print('Error: Failed to register advertisement: ' + str(error))
    mainloop.quit()

def register_app_cb():
    print('GATT application registered')

def register_app_error_cb(error):
    print('Failed to register application: ' + str(error))
    mainloop.quit()

def set_connected_status(status):
    if (status == 1):
        print("connected")
        connected = 1
        stop_advertising()
    else:
        print("disconnected")
        connected = 0
        start_advertising()

def properties_changed(interface, changed, invalidated, path):
    if (interface == bluetooth_constants.DEVICE_INTERFACE):
        if ("Connected" in changed):
            set_connected_status(changed["Connected"])

def interfaces_added(path, interfaces):
    if bluetooth_constants.DEVICE_INTERFACE in interfaces:
        properties = interfaces[bluetooth_constants.DEVICE_INTERFACE]
        if ("Connected" in properties):
            set_connected_status(properties["Connected"])

def stop_advertising():
    global adv
    global adv_mgr_interface
    print("Unregistering advertisement",adv.get_path())
    adv_mgr_interface.UnregisterAdvertisement(adv.get_path())

def start_advertising():
    global adv
    global adv_mgr_interface
    # we're only registering one advertisement object so index (arg2) is hard coded as 0
    print("Registering advertisement",adv.get_path())
    adv_mgr_interface.RegisterAdvertisement(adv.get_path(), {},
                                        reply_handler=register_ad_cb,
                                        error_handler=register_ad_error_cb)
            
dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
bus = dbus.SystemBus()
# we're assuming the adapter supports advertising
adapter_path = bluetooth_constants.BLUEZ_NAMESPACE + bluetooth_constants.ADAPTER_NAME
adv_mgr_interface = dbus.Interface(bus.get_object(bluetooth_constants.BLUEZ_SERVICE_NAME,adapter_path), bluetooth_constants.ADVERTISING_MANAGER_INTERFACE)

bus.add_signal_receiver(properties_changed,
        dbus_interface = bluetooth_constants.DBUS_PROPERTIES,
        signal_name = "PropertiesChanged",
        path_keyword = "path")

bus.add_signal_receiver(interfaces_added,
        dbus_interface = bluetooth_constants.DBUS_OM_IFACE,
        signal_name = "InterfacesAdded")


# we're only registering one advertisement object so index (arg2) is hard coded as 0
adv_mgr_interface = dbus.Interface(bus.get_object(bluetooth_constants.BLUEZ_SERVICE_NAME,adapter_path), bluetooth_constants.ADVERTISING_MANAGER_INTERFACE)
adv = Advertisement(bus, 0, 'peripheral')
start_advertising()

mainloop = GLib.MainLoop()

app = Application(bus)

print('Registering GATT application...')

service_manager = dbus.Interface(
        bus.get_object(bluetooth_constants.BLUEZ_SERVICE_NAME, adapter_path),
        bluetooth_constants.GATT_MANAGER_INTERFACE)

service_manager.RegisterApplication(app.get_path(), {},
                                reply_handler=register_app_cb,
                                error_handler=register_app_error_cb)
mainloop.run()
