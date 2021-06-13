import time
import board
import adafruit_dht
import RPi.GPIO as GPIO
import requests
import json

dhtDevice = adafruit_dht.DHT11(board.D4)

MOISTURE_PIN = 14
LIGHT_PIN = 17
BACKEND_URL = "https://pixmo.herokuapp.com/input"

GPIO.setmode(GPIO.BCM)
GPIO.setup(LIGHT_PIN, GPIO.IN)
GPIO.setup(MOISTURE_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

while True:
    try:
        temperature_c = dhtDevice.temperature
        temperature_f = temperature_c * (9 / 5) + 32
        humidity = dhtDevice.humidity
        is_dark = GPIO.input(LIGHT_PIN)
        is_dry = GPIO.input(MOISTURE_PIN)

        data = {
            "username": "akhilmhdh",
            "password": "akhilmhdh@9",
            "data": {
                "temperature": temperature_c,
                "humidity": humidity,
                "moisture": is_dry,
                "light": is_dark,
            },
        }

        req = requests.post(BACKEND_URL, json=data)
        print(req.status_code)

        light = "dark" if is_dark else "bright"
        moisture = "need water" if is_dry else "water rich"

        print(
            "Temp: {:.1f} F / {:.1f} C    Humidity: {}%   Environment: {}   Water: {}".format(
                temperature_f, temperature_c, humidity, light, moisture
            )
        )

    except RuntimeError as error:
        # Errors happen fairly often, DHT's are hard to read, just keep going
        time.sleep(2.0)
        continue
    except Exception as error:
        dhtDevice.exit()
        raise error

    time.sleep(2)
