import time
import board
import adafruit_dht
import RPi.GPIO as GPIO

dhtDevice = adafruit_dht.DHT11(board.D4)
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN)

while True:
    try:
        temperature_c = dhtDevice.temperature
        temperature_f = temperature_c * (9 / 5) + 32
        humidity = dhtDevice.humidity
        if GPIO.input(17):
            light = "bright"
        else:
            light = "dark"
        print(
            "Temp: {:.1f} F / {:.1f} C    Humidity: {}%   Environment: {}".format(
                temperature_f, temperature_c, humidity, light
            )
        )

    except RuntimeError as error:
        # Errors happen fairly often, DHT's are hard to read, just keep going
        print(error.args[0])
        time.sleep(2.0)
        continue
    except Exception as error:
        dhtDevice.exit()
        raise error

    time.sleep(2)
