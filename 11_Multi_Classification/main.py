from manim import *
import numpy as np

config.disable_caching = True

def get_force_field_func(*point_strength_pairs, **kwargs):
    """
    Creates a force field function from point charges.
    
    Args:
        point_strength_pairs: Tuples of (position, strength)
        radius: Minimum radius for force calculation
    """
    radius = kwargs.get("radius", 0.5)

    def func(point):
        result = np.array([0.0, 0.0, 0.0])
        for center, strength in point_strength_pairs:
            to_center = center - point
            norm = np.linalg.norm(to_center)
            if norm == 0:
                continue
            elif norm < radius:
                to_center /= radius**3
            elif norm >= radius:
                to_center /= norm**3
            to_center *= -strength
            result += to_center
        return result
    return func


class Scene1(Scene):
    """
    Magnetic field animation scene based on div_curl.py
    """
    def construct(self):
        # Show magnetic field only
        self.show_magnetic_force()

    def show_magnetic_force(self):
        """Show magnetic field around a dipole"""
        # Create magnetic force field function
        # Two opposite charges to simulate magnetic dipole
        magnetic_func = get_force_field_func(
            (3 * LEFT, -1), (3 * RIGHT, +1)
        )
        
        # Create vector field
        magnetic_field = ArrowVectorField(
            magnetic_func,
            x_range=[-8, 8, 0.5],
            y_range=[-4, 4, 0.5],
            length_func=lambda norm: 0.4 * np.tanh(norm)
        )
        
        # Animate the magnetic field
        self.play(
            Create(magnetic_field),
            run_time=3
        )
        
        # Store vector field for potential further use
        self.vector_field = magnetic_field
        
        # Create and start stream lines animation directly
        stream_lines = StreamLines(
            magnetic_func,
            stroke_width=2,
            max_anchors_per_line=30,
        )
        
        # Start animating the stream lines immediately
        self.add(stream_lines)
        stream_lines.start_animation(warm_up=True, flow_speed=1.5)
        
        # Wait a moment to establish the field
        self.wait(2)
        
        # Add the spin-1 particle
        self.add_spin_1_particle(magnetic_func)
        
        # Continue showing field interaction
        self.wait(3)
        
        # Stop the stream animation
        stream_lines.end_animation()

    def add_spin_1_particle(self, magnetic_func):
        """Add a spin-1 massive particle and show its behavior in the magnetic field"""
        
        # Create the particle (larger circle to show it's massive)
        particle = Circle(radius=0.3, color=YELLOW, fill_opacity=0.8, stroke_width=3)
        
        # Add glow effect to make it stand out
        glow = Circle(radius=0.5, color=YELLOW, fill_opacity=0.2, stroke_opacity=0)
        
        # Spin-1 representation: Three spin vectors (triplet state)
        spin_arrows = VGroup()
        for i, angle in enumerate([0, 120, 240]):
            start_point = 0.1 * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0])
            end_point = 0.4 * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0])
            arrow = Arrow(
                start=start_point,
                end=end_point,
                color=RED,
                stroke_width=4,
                tip_length=0.15,
                max_stroke_width_to_length_ratio=10
            )
            spin_arrows.add(arrow)
        
        # Group particle components
        particle_group = VGroup(glow, particle, spin_arrows)
        
        # Position particle at the center where field is strong
        center_pos = ORIGIN
        particle_group.move_to(center_pos)
        
        # Animate particle appearance
        self.play(
            FadeIn(glow, scale=0.5),
            GrowFromCenter(particle),
            LaggedStart(
                *[Create(arrow) for arrow in spin_arrows],
                lag_ratio=0.3
            ),
            run_time=2
        )
        
        # Show spin precession around the magnetic field direction
        def precess_spins(mob, dt):
            # Get the local magnetic field direction at particle position
            pos = particle.get_center()
            local_B_field = magnetic_func(pos)
            if np.linalg.norm(local_B_field) > 0:
                B_direction = local_B_field / np.linalg.norm(local_B_field)
                # Larmor precession around B field direction
                # Precess at a visible rate for demonstration
                mob.rotate(1.5 * PI * dt, axis=B_direction)
        
        # Start precession animation
        spin_arrows.add_updater(precess_spins)
        
        # Wait to observe the precession
        self.wait(3)
        
        # Show energy level splitting (Zeeman effect)
        energy_levels = VGroup()
        base_energy = 2 * UP + 5 * RIGHT
        
        # Three energy levels for spin-1 (m = -1, 0, +1)
        for i, (m, color) in enumerate([(-1, BLUE), (0, WHITE), (1, RED)]):
            level = Line(LEFT, RIGHT, color=color, stroke_width=4)
            level.move_to(base_energy + i * 0.3 * UP)
            
            # Add quantum number label
            label = MathTex(f"m = {m:+d}" if m != 0 else "m = 0", font_size=24)
            label.next_to(level, RIGHT, buff=0.2)
            label.set_color(color)
            
            energy_levels.add(VGroup(level, label))
        
        # Animate energy level splitting
        self.play(
            LaggedStart(
                *[DrawBorderThenFill(level) for level in energy_levels],
                lag_ratio=0.2
            ),
            run_time=2
        )
        
        # Final wait with all effects
        self.wait(2)
        
        # Clean up updaters
        spin_arrows.clear_updaters()


class SpinIsIntrinsic(Scene):
    """
    Inspired by `_2018/quaternions.py::QuantumSpin`, this scene clarifies that
    quantum "spin" is intrinsic (built-in), not literal spinning of a particle.
    It sequentially presents the three sentences with simple visuals:
    - Cross out a literal-rotation depiction
    - Show an internal angular-momentum vector inside the particle
    - Highlight the statement "spin = 1" as an intrinsic amount
    """
    def construct(self):
        # Title
        title = Title("Spin is intrinsic")
        self.play(Write(title))
        self.wait(0.5)

        # Base particle
        particle = Circle(radius=0.5, color=BLUE, stroke_width=4)
        particle.set_fill(BLUE, opacity=0.2)
        particle.move_to(ORIGIN)
        core = Dot(ORIGIN, color=BLUE, radius=0.06)

        # --- 1) Not literal rotation ---
        # Curved arcs suggesting rotation around the particle
        arc1 = Arc(start_angle=10 * DEGREES, angle=150 * DEGREES, radius=0.8, color=GREY_B)
        arc2 = Arc(start_angle=200 * DEGREES, angle=150 * DEGREES, radius=0.8, color=GREY_B)
        arcs = VGroup(arc1, arc2)

        # A red "X" to negate the idea of literal spinning
        cross = VGroup(
            Line(0.9 * UL, 0.9 * DR),
            Line(0.9 * UR, 0.9 * DL),
        ).set_stroke(RED, width=6)

        line1 = Text(
            "Even though the word \"spin\" sounds like the particle is physically rotating, it's not.",
            font_size=32,
        )
        line1.to_edge(DOWN)

        self.play(FadeIn(particle), FadeIn(core))
        self.play(Create(arcs))
        self.play(Write(line1))
        self.wait(0.5)
        self.play(Create(cross))
        self.wait(1.0)

        # Transition out the negated depiction
        self.play(FadeOut(VGroup(arcs, cross), scale=0.9))

        # --- 2) Intrinsic angular momentum ---
        # An internal vector representing intrinsic angular momentum
        spin_vec = Arrow(ORIGIN, 0.7 * UP, color=YELLOW, buff=0, stroke_width=6)
        spin_vec.move_to(particle.get_center())

        intrinsic_label = Text("intrinsic angular momentum", font_size=28, color=YELLOW)
        intrinsic_label.next_to(particle, RIGHT, buff=0.6)
        brace = BraceBetweenPoints(particle.get_right(), particle.get_left(), direction=UP)
        brace_label = Text("built into the particle", font_size=26)
        brace_label.next_to(brace, UP, buff=0.2)

        # Mass/charge icons to compare "intrinsic" attributes
        tag_group = VGroup(
            self._tag_circle("m"),
            self._tag_circle("q"),
        ).arrange(RIGHT, buff=0.2).next_to(particle, LEFT, buff=0.6)

        line2 = Text(
            "Instead, spin is an intrinsic form of angular momentum—",
            font_size=30,
        )
        line2b = Text(
            "something built into the particle itself, just like its mass or charge.",
            font_size=30,
        )
        lines2 = VGroup(line2, line2b).arrange(DOWN, aligned_edge=LEFT)
        lines2.to_edge(DOWN)

        self.play(
            GrowArrow(spin_vec),
            FadeIn(intrinsic_label, shift=RIGHT * 0.2),
            GrowFromCenter(brace),
            FadeIn(brace_label, shift=UP * 0.1),
        )
        self.play(FadeIn(tag_group, shift=LEFT * 0.2))
        self.play(ReplacementTransform(line1, lines2))
        self.wait(1.0)

        # --- 3) Spin equals one (an intrinsic amount) ---
        s_eq_one = MathTex("s = 1").scale(1.1).set_color(YELLOW)
        s_eq_one.next_to(particle, UP, buff=0.8)

        line3 = Text(
            "So when we say spin equal to one, we're talking about how much",
            font_size=28,
        )
        line3b = Text(
            "intrinsic angular momentum that particle carries.",
            font_size=28,
        )
        lines3 = VGroup(line3, line3b).arrange(DOWN, aligned_edge=LEFT)
        lines3.to_edge(DOWN)

        self.play(Write(s_eq_one))
        self.play(ReplacementTransform(lines2, lines3))
        self.wait(1.5)

        # Small polish: a gentle precession of the internal vector (not the particle itself)
        def precess(mob, dt):
            mob.rotate(0.8 * dt, axis=OUT, about_point=particle.get_center())

        spin_vec.add_updater(precess)
        self.wait(2.0)
        spin_vec.clear_updaters()

        # Closing beat
        self.play(*map(FadeOut, [lines3, s_eq_one, intrinsic_label, brace, brace_label, tag_group]))
        self.play(FadeOut(VGroup(particle, core, spin_vec)))
        self.wait(0.3)

    def _tag_circle(self, text_str: str) -> VGroup:
        """Small labeled circle tag (for m, q)."""
        circ = Circle(radius=0.18, color=GREY_B, stroke_width=2)
        circ.set_fill(GREY_D, opacity=0.2)
        label = Text(text_str, font_size=24)
        tag = VGroup(circ, label)
        label.move_to(circ.get_center())
        return tag






class QuantumSpin(Scene):
    def construct(self):
        title = Title("Two-state system")

        electron = Dot(color=BLUE)
        electron.set_height(1)
        electron.set_sheen(0.3, UL)
        electron.set_fill(opacity=0.8)
        kwargs = {
            "path_arc": PI,
        }
        curved_arrows = VGroup(
            Arrow(RIGHT, LEFT, **kwargs),
            Arrow(LEFT, RIGHT, **kwargs),
        )
        curved_arrows.set_color(GREY_B)
        curved_arrows.set_stroke(width=2)

        y_arrow = Vector(UP)
        y_arrow.set_color(RED)

        y_ca = curved_arrows.copy()
        y_ca.rotate(70 * DEGREES, LEFT)
        y_group = VGroup(y_ca[0], y_arrow, electron.copy(), y_ca[1])

        x_arrow = y_arrow.copy().rotate(90 * DEGREES, IN, about_point=ORIGIN)
        x_arrow.set_color(GREEN)
        x_ca = curved_arrows.copy()
        x_ca.rotate(70 * DEGREES, UP)
        x_group = VGroup(x_ca[0], x_arrow, electron.copy(), x_ca[1])

        z_ca = curved_arrows.copy()
        z_group = VGroup(electron.copy(), z_ca, Dot(color=BLUE))

        groups = VGroup(x_group, y_group, z_group)
        groups.arrange(RIGHT, buff=1.5)
        groups.move_to(UP)

        y_ca.rotation = Rotating(
            y_ca, axis=rotate_vector(OUT, 70 * DEGREES, LEFT)
        )
        x_ca.rotation = Rotating(
            x_ca, axis=rotate_vector(OUT, 70 * DEGREES, UP)
        )
        z_ca.rotation = Rotating(z_ca, axis=OUT)
        rotations = [ca.rotation for ca in [x_ca, y_ca, z_ca]]

        matrices = VGroup(
            Matrix([["0", "1"], ["1", "0"]]),
            Matrix([["0", "-i"], ["i", "0"]]),
            Matrix([["1", "0"], ["0", "-1"]]),
        )
        for matrix, group in zip(matrices, groups):
            matrix.next_to(group, DOWN)
        for matrix in matrices[1:]:
            matrix.align_to(matrices[0], DOWN)

        self.add(title, groups)
        # Animate matrices appearing with a downward fade-in, while arrows rotate
        self.play(
            LaggedStart(
                *[FadeIn(m, shift=DOWN) for m in matrices],
                lag_ratio=0.2,
                run_time=3,
            ),
            *rotations,
        )
        for x in range(2):
            self.play(*rotations)
