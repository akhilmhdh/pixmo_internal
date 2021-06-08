class SensorReader:
    def __init__(self):
        self.moisture_sensor = 0
        self.temperature_sensor = 0
        self.humdity_sensor = 0
        self.light_sensor = 0

    def read_internal_state(self):
        return True
