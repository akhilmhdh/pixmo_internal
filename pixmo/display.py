import arcade


class Display:
    def __init__(self, width=600, height=600, title="pixmo display"):
        self.width = width
        self.height = height
        self.title = title
        arcade.open_window(self.width, self.height, self.title)
        arcade.set_background_color(arcade.color.WHITE)
        arcade.start_render()

    def draw(self, action, data):
        # Draw the face
        x = 300
        y = 250
        radius = 200
        arcade.draw_circle_filled(x, y, radius, arcade.color.YELLOW)

        x = 370
        y = 350
        # Draw the right eye
        if action == "look":
            x_inc, y_inc = map(int, data.split(":"))
            x += x_inc
            y += y_inc
        radius = 20
        arcade.draw_circle_filled(x, y, radius, arcade.color.BLACK)

        x = 230
        y = 350
        # Draw the left eye
        if action == "look":
            x_inc, y_inc = map(int, data.split(":"))
            x += x_inc
            y += y_inc
        radius = 20
        arcade.draw_circle_filled(x, y, radius, arcade.color.BLACK)

        # Draw the smile 180 -
        x = 300
        y = 220
        tilt = 0

        if action == "sad":
            y = 180
            tilt = 180
        elif action == "person" and data != "happy" and data != "neutral":
            y = 180
            tilt = 180

        width = 120
        height = 100
        start_angle = 190
        end_angle = 350
        arcade.draw_arc_outline(
            x,
            y,
            width,
            height,
            arcade.color.BLACK,
            start_angle,
            end_angle,
            10,
            tilt_angle=tilt,
        )
        # Finish drawing and display the result
        arcade.finish_render()
