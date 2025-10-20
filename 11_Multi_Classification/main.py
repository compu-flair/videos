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

