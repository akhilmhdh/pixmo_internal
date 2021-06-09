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
        arcade.draw_rectangle_filled(
            self.width / 2, self.height / 2, self.width, self.height, arcade.color.WHITE
        )
        x = 300
        y = 300

        text_x_pos = 50
        text_y_pos = 55
        msg = ""

        radius = 200
        arcade.draw_circle_filled(x, y, radius, arcade.color.YELLOW)

        x_inc, y_inc = 0, 0

        x = 370
        y = 350

        # Draw the right eye
        if action == "look":
            x_inc, y_inc = map(int, data.split(":"))
            msg = "What's that"

        radius = 20
        arcade.draw_circle_filled(x + x_inc, y + y_inc, radius, arcade.color.BLACK)

        x = 230
        y = 350

        radius = 20
        arcade.draw_circle_filled(x + x_inc, y + y_inc, radius, arcade.color.BLACK)

        # Draw the smile 180 -
        x = 300
        y = 220
        tilt = 0

        if action == "sad":
            y = 180
            tilt = 180
            msg = "Ohhhh!! I am feeling not okeee"
        elif action == "person" and data != "happy" and data != "neutral":
            y = 180
            tilt = 170
            msg = "It's okay. I am here for you."

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

        arcade.draw_text(
            msg,
            text_x_pos,
            text_y_pos,
            arcade.color.BLACK,
            12,
        )
        # Finish drawing and display the result
        arcade.finish_render()
