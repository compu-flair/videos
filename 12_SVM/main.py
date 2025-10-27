from manim import *
import numpy as np

config.disable_caching = True


class scene1(Scene):
    def construct(self):
        # Set up the pendulum parameters
        pivot_point = np.array([0, 2, 0])
        pendulum_length = 3
        initial_angle = PI / 4  # 45 degrees
        
        # Create the pendulum bob (mass)
        bob_radius = 0.2
        bob = Circle(radius=bob_radius, color=BLUE, fill_opacity=1).set_z_index(10)
        
        # Create the pendulum string
        string = Line(pivot_point, pivot_point + pendulum_length * DOWN, color=WHITE)
        
        # Create middle dotted reference line (equilibrium position)
        middle_line = DashedLine(
            start=pivot_point,
            end=pivot_point + pendulum_length * DOWN,
            color=GRAY,
            stroke_width=2,
            dash_length=0.1
        )
        
        # Create angle sector using always_redraw with custom arc
        def create_angle_arc():
            current_angle = getattr(self, 'current_angle', initial_angle)
            # Create a custom sector by calculating exact points to match the string
            # String uses: bob_x = L*sin(angle), bob_y = -L*cos(angle)
            # We'll build the arc using the EXACT same formula
            
            arc_radius = 1.0
            
            # Create points for the arc using the same trig as the string
            num_arc_points = max(30, int(abs(current_angle) * 60))  # Even more points
            arc_points = []
            
            for i in range(num_arc_points + 1):
                t = i / num_arc_points
                angle_t = t * current_angle
                # Use EXACT same formula as string position
                x = arc_radius * np.sin(angle_t)
                y = -arc_radius * np.cos(angle_t)
                arc_points.append(pivot_point + np.array([x, y, 0]))
            
            # Create the arc as a polyline through these exact points
            arc = VMobject(color=YELLOW, stroke_width=3)
            arc.set_points_as_corners(arc_points)
            # Don't smooth - keep exact points
            
            # Create only line1 (from center to start of arc at vertical down position)
            line1 = Line(pivot_point, arc_points[0], color=YELLOW, stroke_width=3)
            
            # Don't draw line2 - the string itself serves as the second radial line
            # This way there's no confusion or overlap
            
            # Create filled polygon for the sector
            if abs(current_angle) > 0.01:  # Only show sector for significant angles
                sector_points = [pivot_point] + arc_points
                
                sector_fill = Polygon(
                    *sector_points,
                    color=YELLOW,
                    fill_opacity=0.3,
                    stroke_width=0
                )
                
                return VGroup(sector_fill, arc, line1)
            else:
                return VGroup(arc, line1)
        
        angle_arc = always_redraw(create_angle_arc)
        
        # Initialize current angle
        self.current_angle = initial_angle
        
        # Create angle label
        angle_label = MathTex(r"\theta", font_size=36, color=YELLOW)
        
        # Create dynamic labels for Lagrangian and Action
        # Initialize values
        self.kinetic_energy = 0
        self.potential_energy = 0
        self.lagrangian = 0
        self.action = 0
        
        # Create label for Lagrangian: L = T - U with updating decimal
        lagrangian_text = MathTex("L = T - U = ", font_size=28, color=GREEN).to_corner(UL).shift(0.5*DOWN)
        lagrangian_value = DecimalNumber(
            0,
            num_decimal_places=2,
            font_size=28,
            color=GREEN
        )
        lagrangian_value.next_to(lagrangian_text, RIGHT, buff=0.1)
        lagrangian_value.add_updater(lambda m: m.set_value(self.lagrangian))
        
        # Create label for Action: S = ∫L dt with updating decimal
        action_text = MathTex(r"S = \int L\, dt = ", font_size=28, color=ORANGE).to_corner(UL).shift(1.2*DOWN)
        action_value = DecimalNumber(
            0,
            num_decimal_places=2,
            font_size=28,
            color=ORANGE
        )
        action_value.next_to(action_text, RIGHT, buff=0.1)
        action_value.add_updater(lambda m: m.set_value(self.action))
        
        # Function to update pendulum position based on angle
        def update_pendulum(angle):
            # Store current angle for arc updates
            self.current_angle = angle
            
            # Calculate bob position
            bob_x = pendulum_length * np.sin(angle)
            bob_y = -pendulum_length * np.cos(angle)
            bob_pos = pivot_point + np.array([bob_x, bob_y, 0])
            
            # Update bob position
            bob.move_to(bob_pos)
            
            # Update string
            string.put_start_and_end_on(pivot_point, bob_pos)
            
            # Update angle label position
            # Position the label at the end of the arc, slightly offset
            if abs(angle) > 0.1:  # Only show label when angle is significant
                label_radius = 1.5
                label_x = label_radius * np.sin(angle/2)
                label_y = -label_radius * np.cos(angle/2)
                label_pos = pivot_point + np.array([label_x, label_y, 0])
                angle_label.move_to(label_pos)
                angle_label.set_opacity(1)
            else:
                angle_label.set_opacity(0)  # Hide when angle is too small
            
            return bob, string
        
        # Initialize pendulum at starting angle
        update_pendulum(initial_angle)
        
        # Add objects to scene
        self.add(middle_line, angle_arc, bob, string, angle_label, 
                 lagrangian_text, lagrangian_value, action_text, action_value)
        
        # Animate the pendulum swing
        # Using simple harmonic motion approximation for small angles
        swing_duration = 4  # Total duration for one complete cycle
        
        # Create the swinging animation
        def pendulum_updater(mob, dt):
            # Get current time from the scene
            # We'll use a custom time tracker since renderer.time might not be available
            if not hasattr(self, 'custom_time'):
                self.custom_time = 0
            self.custom_time += dt
            
            # Calculate angle using simple harmonic motion
            # θ(t) = θ₀ * cos(ωt) where ω = √(g/L)
            # For animation purposes, we'll use a simplified frequency
            frequency = 2 * PI / swing_duration
            angle = initial_angle * np.cos(frequency * self.custom_time)
            
            # Calculate angular velocity: dθ/dt = -θ₀ * ω * sin(ωt)
            angular_velocity = -initial_angle * frequency * np.sin(frequency * self.custom_time)
            
            # Physics calculations for pendulum
            # Using g = 9.8 m/s^2, L = pendulum_length, m = 1 (unit mass)
            g = 9.8
            m = 1.0
            
            # Kinetic Energy: T = (1/2) * m * L^2 * (dθ/dt)^2
            self.kinetic_energy = 0.5 * m * (pendulum_length ** 2) * (angular_velocity ** 2)
            
            # Potential Energy: U = m * g * L * (1 - cos(θ))
            # Reference at lowest point
            self.potential_energy = m * g * pendulum_length * (1 - np.cos(angle))
            
            # Lagrangian: L = T - U
            self.lagrangian = self.kinetic_energy - self.potential_energy
            
            # Action: S = ∫L dt (numerical integration)
            self.action += self.lagrangian * dt
            
            # Update pendulum position
            update_pendulum(angle)
        
        # Add the updater to the bob (this will update the entire pendulum)
        bob.add_updater(pendulum_updater)
        
        # Let the pendulum swing for several cycles
        self.wait(12)  # 3 complete cycles
        
        # Remove the updater
        bob.remove_updater(pendulum_updater)
        
        self.wait(1)


# Non-optimal pendulum scenes
class scene2(Scene):
    """Pendulum with jerky, non-smooth motion"""
    def construct(self):
        # Set up the pendulum parameters
        pivot_point = np.array([0, 2, 0])
        pendulum_length = 3
        initial_angle = PI / 4
        
        # Create the pendulum bob
        bob_radius = 0.2
        bob = Circle(radius=bob_radius, color=RED, fill_opacity=1).set_z_index(10)
        
        # Create the pendulum string
        string = Line(pivot_point, pivot_point + pendulum_length * DOWN, color=WHITE)
        
        # Create middle dotted reference line
        middle_line = DashedLine(
            start=pivot_point,
            end=pivot_point + pendulum_length * DOWN,
            color=GRAY,
            stroke_width=2,
            dash_length=0.1
        )
        
        # Create angle sector
        def create_angle_arc():
            current_angle = getattr(self, 'current_angle', initial_angle)
            arc_radius = 1.0
            num_arc_points = max(30, int(abs(current_angle) * 60))
            arc_points = []
            
            for i in range(num_arc_points + 1):
                t = i / num_arc_points
                angle_t = t * current_angle
                x = arc_radius * np.sin(angle_t)
                y = -arc_radius * np.cos(angle_t)
                arc_points.append(pivot_point + np.array([x, y, 0]))
            
            arc = VMobject(color=RED, stroke_width=3)
            arc.set_points_as_corners(arc_points)
            line1 = Line(pivot_point, arc_points[0], color=RED, stroke_width=3)
            
            if abs(current_angle) > 0.01:
                sector_points = [pivot_point] + arc_points
                sector_fill = Polygon(*sector_points, color=RED, fill_opacity=0.3, stroke_width=0)
                return VGroup(sector_fill, arc, line1)
            else:
                return VGroup(arc, line1)
        
        angle_arc = always_redraw(create_angle_arc)
        self.current_angle = initial_angle
        
        # Create energy labels
        self.kinetic_energy = 0
        self.potential_energy = 0
        self.lagrangian = 0
        self.action = 0
        
        # Create label for Lagrangian: L = T - U with updating decimal
        lagrangian_text = MathTex("L = T - U = ", font_size=28, color=RED).to_corner(UL).shift(0.5*DOWN)
        lagrangian_value = DecimalNumber(
            0,
            num_decimal_places=2,
            font_size=28,
            color=RED
        )
        lagrangian_value.next_to(lagrangian_text, RIGHT, buff=0.1)
        lagrangian_value.add_updater(lambda m: m.set_value(self.lagrangian))
        
        # Create label for Action: S = ∫L dt with updating decimal
        action_text = MathTex(r"S = \int L\, dt = ", font_size=28, color=ORANGE).to_corner(UL).shift(1.2*DOWN)
        action_value = DecimalNumber(0, num_decimal_places=2, font_size=28, color=ORANGE)
        action_value.next_to(action_text, RIGHT, buff=0.1)
        action_value.add_updater(lambda m: m.set_value(self.action))
        
        # Title
        title = Text("Non-Optimal Path: Straight Line", font_size=32, color=RED)
        title.to_edge(UP)
        
        # Function to update pendulum - modified to allow straight line motion
        def update_pendulum_position(pos):
            bob.move_to(pos)
            string.put_start_and_end_on(pivot_point, pos)
            # Calculate angle from position for the arc display
            dx = pos[0] - pivot_point[0]
            dy = pos[1] - pivot_point[1]
            self.current_angle = np.arctan2(dx, -dy)
        
        # Starting and ending positions
        start_pos = pivot_point + np.array([
            pendulum_length * np.sin(initial_angle),
            -pendulum_length * np.cos(initial_angle),
            0
        ])
        end_pos = pivot_point + np.array([
            pendulum_length * np.sin(-initial_angle),
            -pendulum_length * np.cos(-initial_angle),
            0
        ])
        
        update_pendulum_position(start_pos)
        
        self.add(title, middle_line, angle_arc, bob, string, 
                 lagrangian_text, lagrangian_value, action_text, action_value)
        
        # Non-optimal motion: straight line path instead of circular arc
        def straight_path_updater(mob, dt):
            if not hasattr(self, 'custom_time'):
                self.custom_time = 0
            self.custom_time += dt
            
            # Move in a straight line back and forth
            total_duration = 4.0
            half_duration = total_duration / 2
            t_in_cycle = self.custom_time % total_duration
            
            if t_in_cycle < half_duration:
                # Going from start to end position
                t_normalized = t_in_cycle / half_duration
                current_pos = start_pos + t_normalized * (end_pos - start_pos)
                velocity = (end_pos - start_pos) / half_duration
            else:
                # Going from end back to start position
                t_normalized = (t_in_cycle - half_duration) / half_duration
                current_pos = end_pos + t_normalized * (start_pos - end_pos)
                velocity = (start_pos - end_pos) / half_duration
            
            # Update position
            update_pendulum_position(current_pos)
            
            # Calculate physics for straight line motion
            # The bob moves in a straight line, not following circular constraint
            velocity_magnitude = np.linalg.norm(velocity)
            
            g = 9.8
            m = 1.0
            # Kinetic energy from linear motion
            self.kinetic_energy = 0.5 * m * (velocity_magnitude ** 2)
            
            # Potential energy based on height
            height = current_pos[1] - (pivot_point[1] - pendulum_length)
            self.potential_energy = m * g * height
            
            self.lagrangian = self.kinetic_energy - self.potential_energy
            self.action += self.lagrangian * dt
        
        bob.add_updater(straight_path_updater)
        self.wait(12)  # Three full cycles
        bob.remove_updater(straight_path_updater)
        self.wait(1)


class scene3(Scene):
    """Pendulum taking the upper path - a truly non-optimal route"""
    def construct(self):
        pivot_point = np.array([0, -0.5, 0])
        pendulum_length = 3
        initial_angle = PI / 4  # Starting at 45 degrees to the right
        
        bob_radius = 0.2
        bob = Circle(radius=bob_radius, color=ORANGE, fill_opacity=1).set_z_index(10)
        string = Line(pivot_point, pivot_point + pendulum_length * DOWN, color=WHITE)
        
        middle_line = DashedLine(
            start=pivot_point,
            end=pivot_point + pendulum_length * DOWN,
            color=GRAY,
            stroke_width=2,
            dash_length=0.1
        )
        
        def create_angle_arc():
            current_angle = getattr(self, 'current_angle', initial_angle)
            arc_radius = 1.0
            num_arc_points = max(30, int(abs(current_angle) * 60))
            arc_points = []
            
            for i in range(num_arc_points + 1):
                t = i / num_arc_points
                angle_t = t * current_angle
                x = arc_radius * np.sin(angle_t)
                y = -arc_radius * np.cos(angle_t)
                arc_points.append(pivot_point + np.array([x, y, 0]))
            
            arc = VMobject(color=ORANGE, stroke_width=3)
            arc.set_points_as_corners(arc_points)
            line1 = Line(pivot_point, arc_points[0], color=ORANGE, stroke_width=3)
            
            if abs(current_angle) > 0.01:
                sector_points = [pivot_point] + arc_points
                sector_fill = Polygon(*sector_points, color=ORANGE, fill_opacity=0.3, stroke_width=0)
                return VGroup(sector_fill, arc, line1)
            else:
                return VGroup(arc, line1)
        
        angle_arc = always_redraw(create_angle_arc)
        self.current_angle = initial_angle
        
        # Create energy labels
        self.kinetic_energy = 0
        self.potential_energy = 0
        self.lagrangian = 0
        self.action = 0
        
        # Create label for Lagrangian: L = T - U with updating decimal
        lagrangian_text = MathTex("L = T - U = ", font_size=28, color=ORANGE).to_corner(UL).shift(0.5*DOWN)
        lagrangian_value = DecimalNumber(
            0,
            num_decimal_places=2,
            font_size=28,
            color=ORANGE
        )
        lagrangian_value.next_to(lagrangian_text, RIGHT, buff=0.1)
        lagrangian_value.add_updater(lambda m: m.set_value(self.lagrangian))
        
        # Create label for Action: S = ∫L dt with updating decimal
        action_text = MathTex(r"S = \int L\, dt = ", font_size=28, color=YELLOW).to_corner(UL).shift(1.2*DOWN)
        action_value = DecimalNumber(0, num_decimal_places=2, font_size=28, color=YELLOW)
        action_value.next_to(action_text, RIGHT, buff=0.1)
        action_value.add_updater(lambda m: m.set_value(self.action))
        
        title = Text("Non-Optimal Path: Upper Route", font_size=32, color=ORANGE)
        title.to_edge(UP)
        
        def update_pendulum(angle):
            self.current_angle = angle
            bob_x = pendulum_length * np.sin(angle)
            bob_y = -pendulum_length * np.cos(angle)
            bob_pos = pivot_point + np.array([bob_x, bob_y, 0])
            bob.move_to(bob_pos)
            string.put_start_and_end_on(pivot_point, bob_pos)
            return bob, string
        
        update_pendulum(initial_angle)
        self.add(title, middle_line, angle_arc, bob, string, 
                 lagrangian_text, lagrangian_value, action_text, action_value)
        
        # Motion that goes the upper way (through the top of the circle)
        # This is highly non-optimal as it requires going against gravity
        def upper_path_updater(mob, dt):
            if not hasattr(self, 'custom_time'):
                self.custom_time = 0
            self.custom_time += dt
            
            # Move from +PI/4 to -PI/4 through upper route, then back the same way
            total_duration = 8.0
            half_duration = total_duration / 2
            t_in_cycle = self.custom_time % total_duration
            
            start_angle = PI / 4
            end_angle = -PI / 4
            angle_range = 3 * PI / 2  # Total angle to sweep through upper semicircle
            
            if t_in_cycle < half_duration:
                # Going forward: from +PI/4 to -PI/4 through the top
                t_normalized = t_in_cycle / half_duration
                angle = start_angle + t_normalized * angle_range
                angular_velocity = angle_range / half_duration
            else:
                # Going backward: from -PI/4 back to +PI/4 through the top (reverse)
                t_normalized = (t_in_cycle - half_duration) / half_duration
                angle = (start_angle + angle_range) - t_normalized * angle_range
                angular_velocity = -angle_range / half_duration
            
            g = 9.8
            m = 1.0
            self.kinetic_energy = 0.5 * m * (pendulum_length ** 2) * (angular_velocity ** 2)
            self.potential_energy = m * g * pendulum_length * (1 - np.cos(angle))
            self.lagrangian = self.kinetic_energy - self.potential_energy
            self.action += self.lagrangian * dt
            
            update_pendulum(angle)
        
        bob.add_updater(upper_path_updater)
        self.wait(16)  # Two full back-and-forth cycles
        bob.remove_updater(upper_path_updater)
        self.wait(1)


class SVM_2D_Intro(Scene):
    """2D visualization of SVM with two features"""
    def construct(self):
        # Generate sample data points first
        np.random.seed(42)
        
        # Class -1 (RED) - centered around (-1.5, -1.5)
        n_points_per_class = 15
        class_neg_x1 = np.random.randn(n_points_per_class) * 0.8 - 1.5
        class_neg_x2 = np.random.randn(n_points_per_class) * 0.8 - 1.5
        
        # Class +1 (BLUE) - centered around (1.5, 1.5)
        class_pos_x1 = np.random.randn(n_points_per_class) * 0.8 + 1.5
        class_pos_x2 = np.random.randn(n_points_per_class) * 0.8 + 1.5
        
        # Create dataset table
        from manim import Table
        
        # Prepare table data (9 rows + vdots row)
        table_data = []
        for i in range(5):
            table_data.append([
                f"{class_neg_x1[i]:.1f}",
                f"{class_neg_x2[i]:.1f}",
                "-1"
            ])
        
        for i in range(4):
            table_data.append([
                f"{class_pos_x1[i]:.1f}",
                f"{class_pos_x2[i]:.1f}",
                "+1"
            ])
        
        # Add vdots row using string representation
        table_data.append([
            "⋮",
            "⋮",
            "⋮"
        ])
        
        # Create table with column headers
        dataset = Table(
            table_data,
            col_labels=[MathTex("x_1"), MathTex("x_2"), MathTex("y")],
            include_outer_lines=True,
            element_to_mobject_config={"font_size": 28},
            line_config={"color": YELLOW}
        ).scale(0.6)
        
        # Color the header row
        dataset.get_entries((1, 1)).set_color(GREEN)  # x1 header
        dataset.get_entries((1, 2)).set_color(GREEN)  # x2 header
        dataset.get_entries((1, 3)).set_color(GREEN)  # y header
        
        # Color the data rows
        for i in range(5):
            dataset.get_entries((i+2, 1)).set_color(RED)
            dataset.get_entries((i+2, 2)).set_color(RED)
            dataset.get_entries((i+2, 3)).set_color(RED)
        
        for i in range(5, 9):
            dataset.get_entries((i+2, 1)).set_color(BLUE)
            dataset.get_entries((i+2, 2)).set_color(BLUE)
            dataset.get_entries((i+2, 3)).set_color(BLUE)
        
        # Color and enlarge the vdots row (row 11)
        dataset.get_entries((11, 1)).set_color(ORANGE).scale(2)
        dataset.get_entries((11, 2)).set_color(ORANGE).scale(2)
        dataset.get_entries((11, 3)).set_color(ORANGE).scale(2)
        
        # Show dataset
        self.play(Create(dataset))
        self.wait(1)
        
        # Set up 2D axes - scaled down
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=6,
            y_length=6,
            axis_config={"include_tip": True, "include_numbers": False},
        )
        
        # Add axis labels
        axis_x_label = MathTex("x_1", font_size=28).next_to(axes.x_axis, RIGHT)
        axis_y_label = MathTex("x_2", font_size=28).next_to(axes.y_axis, UP)
        
        # Move dataset to the left side and show axes on the right
        self.play(
            dataset.animate.scale(0.8).to_edge(LEFT, buff=0.3),
            run_time=1.5
        )
        self.play(Create(axes), Write(axis_x_label), Write(axis_y_label), run_time=1.5)
        self.wait()
        
        # Create target dots on the graph
        red_points = VGroup()
        blue_points = VGroup()
        
        for i in range(n_points_per_class):
            red_dot = Dot(
                point=axes.c2p(class_neg_x1[i], class_neg_x2[i]),
                radius=0.08,
                color=RED
            )
            red_points.add(red_dot)
            
            blue_dot = Dot(
                point=axes.c2p(class_pos_x1[i], class_pos_x2[i]),
                radius=0.08,
                color=BLUE
            )
            blue_points.add(blue_dot)
        
        # Extract individual row entries from table (skip header row, skip vdots row)
        table_rows = []
        for i in range(9):  # Only 9 data rows (not including vdots)
            # Get the row entries (row index is i+2 because row 1 is header)
            row_group = VGroup(
                dataset.get_entries((i+2, 1)),
                dataset.get_entries((i+2, 2)),
                dataset.get_entries((i+2, 3))
            )
            table_rows.append(row_group)
        
        # Get the vdots row
        vdots_row = VGroup(
            dataset.get_entries((11, 1)),
            dataset.get_entries((11, 2)),
            dataset.get_entries((11, 3))
        )
        
        # Animate table rows transforming to dots on the graph
        # First 5 rows (red) transform to first 5 red points
        transforms = []
        for i in range(5):
            transforms.append(Transform(table_rows[i], red_points[i]))
        
        # Next 4 rows (blue) transform to first 4 blue points
        for i in range(4):
            transforms.append(Transform(table_rows[5+i], blue_points[i]))
        
        # Transform vdots to all remaining points
        remaining_points = VGroup(
            *[red_points[i] for i in range(5, n_points_per_class)],
            *[blue_points[i] for i in range(4, n_points_per_class)]
        )
        transforms.append(Transform(vdots_row, remaining_points))
        
        # Fade out table frame and header while transforming rows
        self.play(
            *transforms,
            FadeOut(dataset.get_horizontal_lines()),
            FadeOut(dataset.get_vertical_lines()),
            FadeOut(dataset.get_entries((1, 1))),  # x1 header
            FadeOut(dataset.get_entries((1, 2))),  # x2 header
            FadeOut(dataset.get_entries((1, 3))),  # y header
            run_time=2
        )
        self.wait(0.5)
        
        # Directly show optimal boundary (x1 + x2 = 0, or x2 = -x1)
        # This corresponds to w^T x + b = 0, where w = [1, 1]^T and b = 0
        optimal_line = axes.plot(
            lambda x: -x,
            x_range=[-3.5, 3.5],
            color=YELLOW,
            stroke_width=3  # Slightly thinner
        )
        
    
        
        self.play(Create(optimal_line), run_time=2)
        self.wait(1)
        
        # Identify support vectors (points closest to the decision boundary)
        # For decision boundary x1 + x2 = 0, find points closest to this line
        
        # Calculate distances from red points to boundary
        red_distances = []
        for i in range(n_points_per_class):
            # Distance from point (x1, x2) to line x1 + x2 = 0 is |x1 + x2| / sqrt(2)
            dist = abs(class_neg_x1[i] + class_neg_x2[i]) / np.sqrt(2)
            red_distances.append((dist, i))
        
        # Calculate distances from blue points to boundary
        blue_distances = []
        for i in range(n_points_per_class):
            dist = abs(class_pos_x1[i] + class_pos_x2[i]) / np.sqrt(2)
            blue_distances.append((dist, i))
        
        # Sort and find closest points (support vectors)
        red_distances.sort()
        blue_distances.sort()
        
        # Get indices of 2-3 closest points from each class
        num_support_vectors = 3
        red_sv_indices = [idx for _, idx in red_distances[:num_support_vectors]]
        blue_sv_indices = [idx for _, idx in blue_distances[:num_support_vectors]]
        
        # Create dotted lines from support vectors to decision boundary
        support_vector_lines = VGroup()
        
        for idx in red_sv_indices:
            # Calculate perpendicular distance to line x1 + x2 = 0
            x1, x2 = class_neg_x1[idx], class_neg_x2[idx]
            # Closest point on line: project point onto line
            t = (x1 + x2) / 2
            closest_x1 = x1 - t
            closest_x2 = x2 - t
            
            dashed_line = DashedLine(
                start=axes.c2p(x1, x2),
                end=axes.c2p(closest_x1, closest_x2),
                color=WHITE,
                stroke_width=2,
                dash_length=0.1
            )
            support_vector_lines.add(dashed_line)
        
        for idx in blue_sv_indices:
            x1, x2 = class_pos_x1[idx], class_pos_x2[idx]
            t = (x1 + x2) / 2
            closest_x1 = x1 - t
            closest_x2 = x2 - t
            
            dashed_line = DashedLine(
                start=axes.c2p(x1, x2),
                end=axes.c2p(closest_x1, closest_x2),
                color=WHITE,
                stroke_width=2,
                dash_length=0.1
            )
            support_vector_lines.add(dashed_line)
        
        # Animate support vector lines appearing
        self.play(
            LaggedStart(*[Create(line) for line in support_vector_lines], lag_ratio=0.1),
            run_time=2
        )
        self.wait(1)
        
        # Create margin boundaries (parallel lines on either side)
        # For x1 + x2 = 0, margin lines are x1 + x2 = ±margin_distance
        # Calculate approximate margin from support vectors
        avg_red_dist = np.mean([abs(class_neg_x1[idx] + class_neg_x2[idx]) for idx in red_sv_indices])
        avg_blue_dist = np.mean([abs(class_pos_x1[idx] + class_pos_x2[idx]) for idx in blue_sv_indices])
        margin = (avg_red_dist + avg_blue_dist) / 2
        
        # Create margin lines with lighter color and lower opacity
        margin_line1 = axes.plot(
            lambda x: -x - margin,
            x_range=[-3.5, 3.5],
            color=GREEN,
            stroke_width=2,  # Thinner
            stroke_opacity=0.6
        )
        
        margin_line2 = axes.plot(
            lambda x: -x + margin,
            x_range=[-3.5, 3.5],
            color=GREEN,
            stroke_width=2,  # Thinner
            stroke_opacity=0.6
        )
        
        # Create shaded region for margin
        x_vals = np.linspace(-3.5, 3.5, 50)
        upper_points = [axes.c2p(x, -x + margin) for x in x_vals]
        lower_points = [axes.c2p(x, -x - margin) for x in x_vals[::-1]]
        
        margin_region = Polygon(
            *upper_points,
            *lower_points,
            fill_color=GREEN,
            fill_opacity=0.15,  # More transparent
            stroke_width=0
        )
        
        self.play(
            Create(margin_region),
            Create(margin_line1),
            Create(margin_line2),
            run_time=2
        )
        self.wait(2)
        
        # === MOVE ORIGINAL GRAPH TO LEFT AND ADD TWO MINI GRAPHS ON RIGHT ===
        
        # Group all main graph elements
        main_graph = VGroup(
            axes, axis_x_label, axis_y_label,
            optimal_line, support_vector_lines,
            margin_region, margin_line1, margin_line2
        )
        # Add all the point transforms
        for row in table_rows:
            main_graph.add(row)
        main_graph.add(vdots_row)
        
        # Move main graph to the left
        self.play(
            main_graph.animate.scale(0.85).shift(LEFT * 2.5),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Create two mini windows stacked vertically on the right
        # Top window: w determines orientation (b=0)
        # Bottom window: b determines position (w fixed)
        
        # Top mini graph (w effect)
        top_axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={"include_tip": True, "include_numbers": False, "stroke_width": 2},
        )
        top_axes.to_edge(RIGHT, buff=0.7).shift(UP * 1.5)
        
        top_x_label = MathTex("x_1", font_size=20).next_to(top_axes.x_axis, RIGHT, buff=0.1)
        top_y_label = MathTex("x_2", font_size=20).next_to(top_axes.y_axis, UP, buff=0.1)
        
        # Title for top graph
        top_title = MathTex(r"\mathbf{w} \text{ → orientation}", font_size=24, color=YELLOW)
        top_title.next_to(top_axes, UP, buff=0.2)
        top_subtitle = MathTex(r"b = 0", font_size=20, color=GRAY)
        top_subtitle.next_to(top_title, DOWN, buff=0.1)
        
        # Show 3 different w vectors with different orientations
        w_vectors = [
            (np.array([1, 1]), RED),
            (np.array([1, -1]), BLUE),
            (np.array([2, 1]), GREEN),
        ]
        
        top_lines = VGroup()
        top_arrows = VGroup()
        top_labels = VGroup()
        
        for w, color in w_vectors:
            # Decision boundary line
            if w[1] != 0:
                line = top_axes.plot(
                    lambda x: -(w[0]/w[1]) * x,
                    x_range=[-2.5, 2.5],
                    color=color,
                    stroke_width=2
                )
            else:
                line = Line(
                    top_axes.c2p(0, -2.5),
                    top_axes.c2p(0, 2.5),
                    color=color,
                    stroke_width=2
                )
            
            # w vector (perpendicular to line)
            w_normalized = w / np.linalg.norm(w) * 1.0
            w_arrow = Arrow(
                top_axes.c2p(0, 0),
                top_axes.c2p(w_normalized[0], w_normalized[1]),
                color=color,
                buff=0,
                stroke_width=4,
                max_tip_length_to_length_ratio=0.3
            )
            
            # Small label
            w_label = MathTex(
                f"\\mathbf{{w}}_{{{int(w[0])},{int(w[1])}}}",
                color=color,
                font_size=16
            )
            # Position label
            label_offset = np.array([w_normalized[0]*0.4, w_normalized[1]*0.4, 0])
            w_label.next_to(w_arrow.get_end(), label_offset, buff=0.05)
            
            top_lines.add(line)
            top_arrows.add(w_arrow)
            top_labels.add(w_label)
        
        # Bottom mini graph (b effect)
        bottom_axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={"include_tip": True, "include_numbers": False, "stroke_width": 2},
        )
        bottom_axes.to_edge(RIGHT, buff=0.7).shift(DOWN * 1.5)
        
        bottom_x_label = MathTex("x_1", font_size=20).next_to(bottom_axes.x_axis, RIGHT, buff=0.1)
        bottom_y_label = MathTex("x_2", font_size=20).next_to(bottom_axes.y_axis, UP, buff=0.1)
        
        # Title for bottom graph
        bottom_title = MathTex(r"b \text{ → position}", font_size=24, color=YELLOW)
        bottom_title.next_to(bottom_axes, UP, buff=0.2)
        bottom_subtitle = MathTex(r"\mathbf{w} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}", font_size=18, color=GRAY)
        bottom_subtitle.next_to(bottom_title, DOWN, buff=0.1)
        
        # Show 3 parallel lines with different b values
        w_fixed = np.array([1, 1])
        b_values = [0, -1.2, 1.2]
        colors_b = [YELLOW, ORANGE, PURPLE]
        
        bottom_lines = VGroup()
        bottom_labels = VGroup()
        
        for b, color in zip(b_values, colors_b):
            # Decision boundary: x1 + x2 + b = 0 => x2 = -x1 - b
            line = bottom_axes.plot(
                lambda x, b_val=b: -x - b_val,
                x_range=[-2.5, 2.5],
                color=color,
                stroke_width=2
            )
            
            # Label b value
            b_label = MathTex(f"b={b:.1f}" if b != 0 else "b=0", color=color, font_size=18)
            # Position at the right edge of line
            label_x = 1.5
            label_y = -label_x - b
            if -2.5 <= label_y <= 2.5:
                b_label.move_to(bottom_axes.c2p(label_x, label_y) + RIGHT*0.4)
            else:
                b_label.move_to(bottom_axes.c2p(0, -b) + UP*0.3)
            
            bottom_lines.add(line)
            bottom_labels.add(b_label)
        
        # Show single w vector for reference on bottom graph
        w_normalized_fixed = w_fixed / np.linalg.norm(w_fixed) * 1.0
        w_arrow_fixed = Arrow(
            bottom_axes.c2p(0, 0),
            bottom_axes.c2p(w_normalized_fixed[0], w_normalized_fixed[1]),
            color=WHITE,
            buff=0,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.3
        )
        
        # Add all mini graph elements
        self.add(
            top_axes, top_x_label, top_y_label, top_title, top_subtitle,
            top_lines, top_arrows, top_labels,
            bottom_axes, bottom_x_label, bottom_y_label, bottom_title, bottom_subtitle,
            bottom_lines, bottom_labels, w_arrow_fixed
        )
        
        # Final wait
        self.wait(3)


class SVM_2D_Simple(Scene):
    """Simple 2D visualization of SVM showing optimal separation"""
    def construct(self):
        # Set up 2D axes
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=8,
            y_length=8,
            axis_config={"include_tip": True, "include_numbers": False},
        )
        
        # Add axis labels
        x_label = MathTex("x_1", font_size=36).next_to(axes.x_axis, RIGHT)
        y_label = MathTex("x_2", font_size=36).next_to(axes.y_axis, UP)
        
        self.add(axes, x_label, y_label)
        self.wait(0.5)
        
        # Generate sample data points
        np.random.seed(42)
        
        # Class -1 (RED) - centered around (-1.5, -1.5)
        n_points_per_class = 15
        class_neg_x1 = np.random.randn(n_points_per_class) * 0.8 - 1.5
        class_neg_x2 = np.random.randn(n_points_per_class) * 0.8 - 1.5
        
        # Class +1 (BLUE) - centered around (1.5, 1.5)
        class_pos_x1 = np.random.randn(n_points_per_class) * 0.8 + 1.5
        class_pos_x2 = np.random.randn(n_points_per_class) * 0.8 + 1.5
        
        # Create 2D points as dots
        red_points = VGroup()
        blue_points = VGroup()
        
        for i in range(n_points_per_class):
            # Red points (class -1)
            red_dot = Dot(
                point=axes.c2p(class_neg_x1[i], class_neg_x2[i]),
                radius=0.1,
                color=RED
            )
            red_points.add(red_dot)
            
            # Blue points (class +1)
            blue_dot = Dot(
                point=axes.c2p(class_pos_x1[i], class_pos_x2[i]),
                radius=0.1,
                color=BLUE
            )
            blue_points.add(blue_dot)
        
        # Add title
        title = Text("Support Vector Machine (SVM)", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Animate points appearing
        self.play(
            LaggedStart(*[FadeIn(point, scale=0.5) for point in red_points], lag_ratio=0.05),
            run_time=1.5
        )
        self.play(
            LaggedStart(*[FadeIn(point, scale=0.5) for point in blue_points], lag_ratio=0.05),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Optimal decision boundary: x1 + x2 = 0, or x2 = -x1
        decision_line = axes.plot(
            lambda x: -x,
            x_range=[-3.5, 3.5],
            color=YELLOW,
            stroke_width=4
        )
        
        boundary_label = Text("Decision Boundary", font_size=28, color=YELLOW)
        boundary_label.next_to(title, DOWN, buff=0.3)
        
        self.play(
            Create(decision_line),
            Write(boundary_label),
            run_time=2
        )
        self.wait(1)
        
        # Identify support vectors (points closest to the decision boundary)
        # Calculate distances from red points to boundary
        red_distances = []
        for i in range(n_points_per_class):
            dist = abs(class_neg_x1[i] + class_neg_x2[i]) / np.sqrt(2)
            red_distances.append((dist, i))
        
        # Calculate distances from blue points to boundary
        blue_distances = []
        for i in range(n_points_per_class):
            dist = abs(class_pos_x1[i] + class_pos_x2[i]) / np.sqrt(2)
            blue_distances.append((dist, i))
        
        # Sort and find closest points (support vectors)
        red_distances.sort()
        blue_distances.sort()
        
        # Get indices of 2-3 closest points from each class
        num_support_vectors = 3
        red_sv_indices = [idx for _, idx in red_distances[:num_support_vectors]]
        blue_sv_indices = [idx for _, idx in blue_distances[:num_support_vectors]]
        
        # Create circles around support vectors
        support_vector_circles = VGroup()
        
        for idx in red_sv_indices:
            circle = Circle(
                radius=0.2,
                color=WHITE,
                stroke_width=4
            )
            circle.move_to(axes.c2p(class_neg_x1[idx], class_neg_x2[idx]))
            support_vector_circles.add(circle)
        
        for idx in blue_sv_indices:
            circle = Circle(
                radius=0.2,
                color=WHITE,
                stroke_width=4
            )
            circle.move_to(axes.c2p(class_pos_x1[idx], class_pos_x2[idx]))
            support_vector_circles.add(circle)
        
        # Add support vector label
        sv_label = Text("Support Vectors", font_size=28, color=WHITE)
        sv_label.next_to(boundary_label, DOWN, buff=0.3)
        
        # Animate support vectors appearing
        self.play(
            LaggedStart(*[Create(circle) for circle in support_vector_circles], lag_ratio=0.1),
            Write(sv_label),
            run_time=2
        )
        self.wait(1)
        
        # Create margin boundaries (parallel lines on either side)
        # Calculate approximate margin from support vectors
        avg_red_dist = np.mean([abs(class_neg_x1[idx] + class_neg_x2[idx]) for idx in red_sv_indices])
        avg_blue_dist = np.mean([abs(class_pos_x1[idx] + class_pos_x2[idx]) for idx in blue_sv_indices])
        margin = (avg_red_dist + avg_blue_dist) / 2
        
        # Margin lines: x1 + x2 = ±margin, or x2 = -x1 ± margin
        margin_line1 = axes.plot(
            lambda x: -x - margin,
            x_range=[-3.5, 3.5],
            color=GREEN,
            stroke_width=3,
            stroke_opacity=0.6
        )
        
        margin_line2 = axes.plot(
            lambda x: -x + margin,
            x_range=[-3.5, 3.5],
            color=GREEN,
            stroke_width=3,
            stroke_opacity=0.6
        )
        
        # Create shaded region for margin
        # Get points for the region between the two margin lines
        x_vals = np.linspace(-3.5, 3.5, 50)
        upper_points = [axes.c2p(x, -x + margin) for x in x_vals]
        lower_points = [axes.c2p(x, -x - margin) for x in x_vals[::-1]]
        
        margin_region = Polygon(
            *upper_points,
            *lower_points,
            fill_color=GREEN,
            fill_opacity=0.2,
            stroke_width=0
        )
        
        # Add margin label
        margin_label = Text("Margin", font_size=28, color=GREEN)
        margin_label.next_to(sv_label, DOWN, buff=0.3)
        
        self.play(
            Create(margin_region),
            Create(margin_line1),
            Create(margin_line2),
            Write(margin_label),
            run_time=2
        )
        self.wait(1)
        
        # Add explanation
        explanation = VGroup(
            Text("SVM finds the line that:", font_size=24),
            Text("1. Separates the two classes", font_size=20),
            Text("2. Maximizes the margin", font_size=20),
            Text("3. Is defined by support vectors", font_size=20),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        explanation.to_corner(DR).shift(UP * 0.5)
        
        self.play(Write(explanation), run_time=3)
        self.wait(3)


class scene4(Scene):
    """Combined table view showing energy values for all three pendulum paths"""
    def construct(self):
        # Table parameters
        row_height = 2.5
        col_width = 2.5
        initial_angle = PI / 4
        
        # Create table structure
        # Headers
        headers = VGroup(
            Text("Motion", font_size=20),
            Text("K.E.", font_size=20),
            Text("P.E.", font_size=20),
            Text("L", font_size=20),
            Text("S", font_size=20),
        )
        
        # Position headers
        for i, header in enumerate(headers):
            header.move_to(np.array([col_width * (i - 2), 3.2, 0]))
        
        # Create dividing lines
        h_lines = VGroup()
        for i in range(4):  # 3 rows + 1 top line
            line = Line(
                start=np.array([-col_width * 2.5, 3.5 - i * row_height, 0]),
                end=np.array([col_width * 2.5, 3.5 - i * row_height, 0]),
                color=WHITE,
                stroke_width=2
            )
            h_lines.add(line)
        
        v_lines = VGroup()
        for i in range(6):  # 5 columns + 1 extra
            line = Line(
                start=np.array([col_width * (i - 2.5), 3.5, 0]),
                end=np.array([col_width * (i - 2.5), 3.5 - 3 * row_height, 0]),
                color=WHITE,
                stroke_width=2
            )
            v_lines.add(line)
        
        # Animate table creation
        self.play(Create(h_lines), Create(v_lines), Write(headers), run_time=1.5)
        self.wait(0.5)
        
        # Row labels in the Motion column
        row1_label = Text("Optimal", font_size=18, color=BLUE)
        row1_label.move_to(np.array([col_width * (-2), 3.5 - 0.5 * row_height, 0]))
        
        row2_label = Text("Straight", font_size=18, color=RED)
        row2_label.move_to(np.array([col_width * (-2), 3.5 - 1.5 * row_height, 0]))
        
        row3_label = Text("Upper", font_size=18, color=ORANGE)
        row3_label.move_to(np.array([col_width * (-2), 3.5 - 2.5 * row_height, 0]))
        
        self.play(Write(row1_label), Write(row2_label), Write(row3_label), run_time=1)
        self.wait(0.5)
        
        # Initialize energy variables
        self.ke1 = 0
        self.pe1 = 0
        self.l1 = 0
        self.s1 = 0
        
        self.ke2 = 0
        self.pe2 = 0
        self.l2 = 0
        self.s2 = 0
        
        self.ke3 = 0
        self.pe3 = 0
        self.l3 = 0
        self.s3 = 0
        
        # Row 1: S value only (keep as number)
        s1_val = DecimalNumber(0, num_decimal_places=1, font_size=24, color=RED)
        s1_val.move_to(np.array([col_width * 2, 3.5 - 0.5 * row_height, 0]))
        s1_val.add_updater(lambda m: m.set_value(self.s1))
        
        # Row 2: S value only (keep as number)
        s2_val = DecimalNumber(0, num_decimal_places=1, font_size=24, color=RED)
        s2_val.move_to(np.array([col_width * 2, 3.5 - 1.5 * row_height, 0]))
        s2_val.add_updater(lambda m: m.set_value(self.s2))
        
        # Row 3: S value only (keep as number)
        s3_val = DecimalNumber(0, num_decimal_places=1, font_size=24, color=RED)
        s3_val.move_to(np.array([col_width * 2, 3.5 - 2.5 * row_height, 0]))
        s3_val.add_updater(lambda m: m.set_value(self.s3))
        
        # Add S value displays
        self.add(s1_val, s2_val, s3_val)
        
        # Initialize data tracking lists for curves
        self.ke1_data = []
        self.pe1_data = []
        self.l1_data = []
        self.ke2_data = []
        self.pe2_data = []
        self.l2_data = []
        self.ke3_data = []
        self.pe3_data = []
        self.l3_data = []
        self.time_data = []
        
        # Create axes for plotting energy curves BEFORE the physics updater
        graph_width = 1.8
        graph_height = 1.3
        
        # Helper function to create mini axes
        def create_mini_axes(center_pos):
            axes = Axes(
                x_range=[0, 16, 4],
                y_range=[-3, 3, 1],
                x_length=graph_width,
                y_length=graph_height,
                axis_config={"include_tip": False, "stroke_width": 1},
                x_axis_config={"include_numbers": False},
                y_axis_config={"include_numbers": False},
            )
            axes.move_to(center_pos)
            return axes
        
        # Create axes for each energy type for each path
        # Row 1 (Optimal)
        axes_ke1 = create_mini_axes(np.array([col_width * (-1), 3.5 - 0.5 * row_height, 0]))
        axes_pe1 = create_mini_axes(np.array([col_width * 0, 3.5 - 0.5 * row_height, 0]))
        axes_l1 = create_mini_axes(np.array([col_width * 1, 3.5 - 0.5 * row_height, 0]))
        
        # Row 2 (Straight)
        axes_ke2 = create_mini_axes(np.array([col_width * (-1), 3.5 - 1.5 * row_height, 0]))
        axes_pe2 = create_mini_axes(np.array([col_width * 0, 3.5 - 1.5 * row_height, 0]))
        axes_l2 = create_mini_axes(np.array([col_width * 1, 3.5 - 1.5 * row_height, 0]))
        
        # Row 3 (Upper)
        axes_ke3 = create_mini_axes(np.array([col_width * (-1), 3.5 - 2.5 * row_height, 0]))
        axes_pe3 = create_mini_axes(np.array([col_width * 0, 3.5 - 2.5 * row_height, 0]))
        axes_l3 = create_mini_axes(np.array([col_width * 1, 3.5 - 2.5 * row_height, 0]))
        
        self.add(axes_ke1, axes_pe1, axes_l1)
        self.add(axes_ke2, axes_pe2, axes_l2)
        self.add(axes_ke3, axes_pe3, axes_l3)
        
        # Create curves with always_redraw
        curve_ke1 = always_redraw(lambda: axes_ke1.plot_line_graph(
            self.time_data[-min(len(self.time_data), 200):],
            self.ke1_data[-min(len(self.ke1_data), 200):],
            add_vertex_dots=False,
            line_color=GREEN,
            stroke_width=2
        ) if len(self.time_data) > 1 else VMobject())
        
        curve_pe1 = always_redraw(lambda: axes_pe1.plot_line_graph(
            self.time_data[-min(len(self.time_data), 200):],
            self.pe1_data[-min(len(self.pe1_data), 200):],
            add_vertex_dots=False,
            line_color=YELLOW,
            stroke_width=2
        ) if len(self.time_data) > 1 else VMobject())
        
        curve_l1 = always_redraw(lambda: axes_l1.plot_line_graph(
            self.time_data[-min(len(self.time_data), 200):],
            self.l1_data[-min(len(self.l1_data), 200):],
            add_vertex_dots=False,
            line_color=ORANGE,
            stroke_width=2
        ) if len(self.time_data) > 1 else VMobject())
        
        curve_ke2 = always_redraw(lambda: axes_ke2.plot_line_graph(
            self.time_data[-min(len(self.time_data), 200):],
            self.ke2_data[-min(len(self.ke2_data), 200):],
            add_vertex_dots=False,
            line_color=GREEN,
            stroke_width=2
        ) if len(self.time_data) > 1 else VMobject())
        
        curve_pe2 = always_redraw(lambda: axes_pe2.plot_line_graph(
            self.time_data[-min(len(self.time_data), 200):],
            self.pe2_data[-min(len(self.pe2_data), 200):],
            add_vertex_dots=False,
            line_color=YELLOW,
            stroke_width=2
        ) if len(self.time_data) > 1 else VMobject())
        
        curve_l2 = always_redraw(lambda: axes_l2.plot_line_graph(
            self.time_data[-min(len(self.time_data), 200):],
            self.l2_data[-min(len(self.l2_data), 200):],
            add_vertex_dots=False,
            line_color=ORANGE,
            stroke_width=2
        ) if len(self.time_data) > 1 else VMobject())
        
        curve_ke3 = always_redraw(lambda: axes_ke3.plot_line_graph(
            self.time_data[-min(len(self.time_data), 200):],
            self.ke3_data[-min(len(self.ke3_data), 200):],
            add_vertex_dots=False,
            line_color=GREEN,
            stroke_width=2
        ) if len(self.time_data) > 1 else VMobject())
        
        curve_pe3 = always_redraw(lambda: axes_pe3.plot_line_graph(
            self.time_data[-min(len(self.time_data), 200):],
            self.pe3_data[-min(len(self.pe3_data), 200):],
            add_vertex_dots=False,
            line_color=YELLOW,
            stroke_width=2
        ) if len(self.time_data) > 1 else VMobject())
        
        curve_l3 = always_redraw(lambda: axes_l3.plot_line_graph(
            self.time_data[-min(len(self.time_data), 200):],
            self.l3_data[-min(len(self.l3_data), 200):],
            add_vertex_dots=False,
            line_color=ORANGE,
            stroke_width=2
        ) if len(self.time_data) > 1 else VMobject())
        
        self.add(curve_ke1, curve_pe1, curve_l1)
        self.add(curve_ke2, curve_pe2, curve_l2)
        self.add(curve_ke3, curve_pe3, curve_l3)
        
        # Set constant action values (not changing over time)
        self.s1 = -2.5  # Optimal path has lowest action
        self.s2 = 1.8   # Straight path has higher action
        self.s3 = 5.2   # Upper path has highest action
        
        # Invisible updater to calculate physics (without displaying motion)
        def physics_updater(mob, dt):
            if not hasattr(self, 'custom_time'):
                self.custom_time = 0
            self.custom_time += dt
            
            # Append time
            self.time_data.append(self.custom_time)
            
            g = 9.8
            m = 1.0
            pendulum_length = 1.2
            
            # PATH 1: Optimal circular arc
            frequency = 2 * PI / 4
            angle1 = initial_angle * np.cos(frequency * self.custom_time)
            angular_velocity1 = -initial_angle * frequency * np.sin(frequency * self.custom_time)
            
            self.ke1 = 0.5 * m * (pendulum_length ** 2) * (angular_velocity1 ** 2)
            self.pe1 = m * g * pendulum_length * (1 - np.cos(angle1))
            self.l1 = self.ke1 - self.pe1
            # self.s1 remains constant
            
            self.ke1_data.append(self.ke1)
            self.pe1_data.append(self.pe1)
            self.l1_data.append(self.l1)
            
            # PATH 2: Straight line
            total_duration = 4.0
            half_duration = total_duration / 2
            t_in_cycle = self.custom_time % total_duration
            
            start_x = pendulum_length * np.sin(initial_angle)
            start_y = -pendulum_length * np.cos(initial_angle)
            end_x = pendulum_length * np.sin(-initial_angle)
            end_y = -pendulum_length * np.cos(-initial_angle)
            
            if t_in_cycle < half_duration:
                t_normalized = t_in_cycle / half_duration
                velocity_mag = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2) / half_duration
                current_y = start_y + t_normalized * (end_y - start_y)
            else:
                t_normalized = (t_in_cycle - half_duration) / half_duration
                velocity_mag = np.sqrt((start_x - end_x)**2 + (start_y - end_y)**2) / half_duration
                current_y = end_y + t_normalized * (start_y - end_y)
            
            self.ke2 = 0.5 * m * (velocity_mag ** 2)
            height = current_y - (-pendulum_length)
            self.pe2 = m * g * height
            self.l2 = self.ke2 - self.pe2
            # self.s2 remains constant
            
            self.ke2_data.append(self.ke2)
            self.pe2_data.append(self.pe2)
            self.l2_data.append(self.l2)
            
            # PATH 3: Upper semicircle
            total_duration3 = 8.0
            half_duration3 = total_duration3 / 2
            t_in_cycle3 = self.custom_time % total_duration3
            
            angle_range = 3 * PI / 2
            
            if t_in_cycle3 < half_duration3:
                t_normalized3 = t_in_cycle3 / half_duration3
                angle3 = initial_angle + t_normalized3 * angle_range
                angular_velocity3 = angle_range / half_duration3
            else:
                t_normalized3 = (t_in_cycle3 - half_duration3) / half_duration3
                angle3 = (initial_angle + angle_range) - t_normalized3 * angle_range
                angular_velocity3 = -angle_range / half_duration3
            
            self.ke3 = 0.5 * m * (pendulum_length ** 2) * (angular_velocity3 ** 2)
            self.pe3 = m * g * pendulum_length * (1 - np.cos(angle3))
            self.l3 = self.ke3 - self.pe3
            # self.s3 remains constant
            
            self.ke3_data.append(self.ke3)
            self.pe3_data.append(self.pe3)
            self.l3_data.append(self.l3)
        
        # Create a dummy mob for the updater
        dummy = Dot(fill_opacity=0, stroke_opacity=0)
        self.add(dummy)
        dummy.add_updater(physics_updater)
        
        self.wait(16)
        
        dummy.remove_updater(physics_updater)
        self.wait(1)
