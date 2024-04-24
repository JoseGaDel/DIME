#!/bin/bash
echo "Ensure the device is disconnected, then press enter."
read

ls /dev/tty* > before.txt

echo "Now connect the device and press enter."
read

ls /dev/tty* > after.txt

echo "Changes:"
diff before.txt after.txt | grep '>' | cut -d ' ' -f 2-
