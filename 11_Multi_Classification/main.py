from manim import *
import numpy as np


def get_force_field_func(*point_strength_pairs, **kwargs):
    """
    Generate a force field function based on point charges.
    
    Args:
        point_strength_pairs: Tuples of (center_point, strength)
        radius: Minimum distance to avoid singularities (default 0.5)
    """
    radius = kwargs.get("radius", 0.5)

    def func(point):
        result = np.array([0.0, 0.0, 0.0])
        for center, strength in point_strength_pairs:
            difference = point - center
            distance = np.linalg.norm(difference)
            if distance > radius:
                result += strength * difference / (distance ** 3)
        return result
    return func


def get_charged_particle(color, sign, radius=0.1):
    """Create a charged particle visualization."""
    result = Circle(
        stroke_color=WHITE,
        stroke_width=0.5,
        fill_color=color,
        fill_opacity=0.8,
        radius=radius
    )
    sign_symbol = MathTex(sign)
    sign_symbol.set_stroke(WHITE, 1)
    sign_symbol.set_width(0.5 * result.get_width())
    sign_symbol.move_to(result)
    result.add(sign_symbol)
    return result


class MagneticField(Scene):
    """
    Visualize a magnetic field created by a bar magnet.
    Shows vector field and field lines.
    """
    def construct(self):
        self.add_plane()
        self.show_magnet()
        self.show_vector_field()
        self.show_field_lines()

    def add_plane(self):
        """Add coordinate plane."""
        plane = NumberPlane()
        self.add(plane)
        self.plane = plane

    def show_magnet(self):
        """Create and display a bar magnet."""
        # Create magnet halves
        north_pole = Rectangle(
            width=3.5,
            height=1,
            stroke_width=0,
            fill_opacity=1,
            fill_color=RED
        )
        south_pole = Rectangle(
            width=3.5,
            height=1,
            stroke_width=0,
            fill_opacity=1,
            fill_color=BLUE
        )
        
        # Arrange magnet
        magnet = VGroup(south_pole, north_pole)
        magnet.arrange(RIGHT, buff=0)
        
        # Add labels
        for char, vect in [("S", LEFT), ("N", RIGHT)]:
            label = Text(char)
            label.scale(2)
            label.next_to(magnet, vect)
            label.shift(1.75 * vect)
            magnet.add(label)
        
        self.magnet = magnet
        self.add(magnet)
        self.play(Create(magnet))
        self.wait()

    def show_vector_field(self):
        """Display the magnetic vector field."""
        # Define magnetic field function
        magnetic_func = get_force_field_func(
            (3 * LEFT, -1),  # South pole (attractive)
            (3 * RIGHT, +1)  # North pole (repulsive)
        )
        
        # Create vector field
        vector_field = ArrowVectorField(
            magnetic_func,
            x_range=[-7, 7, 1],
            y_range=[-4, 4, 1],
            length_func=lambda norm: 0.4 * sigmoid(norm)
        )
        
        self.vector_field = vector_field
        self.play(Create(vector_field), run_time=3)
        self.wait()

    def show_field_lines(self):
        """Show magnetic field lines using stream lines."""
        # Create stream lines
        stream_lines = StreamLines(
            self.vector_field.func,
            x_range=[-7, 7, 0.5],
            y_range=[-4, 4, 0.5],
            stroke_width=2,
            max_anchors_per_line=30,
        )
        
        # Animate stream lines
        self.play(stream_lines.create())
        self.wait(2)
        self.play(FadeOut(stream_lines))
        self.wait()


class SimpleMagneticField(Scene):
    """
    Simplified magnetic field visualization for Manim Community Edition.
    """
    def construct(self):
        # Setup plane
        plane = NumberPlane()
        self.add(plane)
        
        # Create magnet
        north_pole = Rectangle(
            width=3.5, height=1,
            stroke_width=2,
            stroke_color=WHITE,
            fill_opacity=0.8,
            fill_color=RED
        )
        south_pole = Rectangle(
            width=3.5, height=1,
            stroke_width=2,
            stroke_color=WHITE,
            fill_opacity=0.8,
            fill_color=BLUE
        )
        
        magnet = VGroup(south_pole, north_pole)
        magnet.arrange(RIGHT, buff=0)
        
        # Add labels
        label_s = Text("S", color=WHITE).scale(1.5)
        label_n = Text("N", color=WHITE).scale(1.5)
        label_s.next_to(south_pole, LEFT).shift(1.5 * LEFT)
        label_n.next_to(north_pole, RIGHT).shift(1.5 * RIGHT)
        
        labels = VGroup(label_s, label_n)
        
        # Magnetic field function
        def magnetic_func(point):
            func = get_force_field_func(
                (3 * LEFT, -1),   # South pole
                (3 * RIGHT, +1)   # North pole
            )
            return func(point)
        
        # Create vector field
        vector_field = ArrowVectorField(
            magnetic_func,
            x_range=[-7, 7, 1],
            y_range=[-4, 4, 1],
            length_func=lambda norm: 0.35 * np.tanh(norm)
        )
        
        # Animate
        self.play(Create(magnet), Write(labels))
        self.wait()
        self.play(Create(vector_field), run_time=2)
        self.wait(2)

