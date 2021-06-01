import RPi.GPIO as GPIO
from time import sleep
MOISTURE_PIN=14

GPIO.setmode(GPIO.BCM)
GPIO.setup(MOISTURE_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

while True:
    if (GPIO.input(MOISTURE_PIN) == True):
        print("I have water")
    else:
        print("I have no water")
    sleep(5)
