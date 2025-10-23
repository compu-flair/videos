from manim import *
import numpy as np
import os
from PIL import Image

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
        self.wait(8)
        
        # Add the spin-1 particle
        self.add_spin_1_particle(magnetic_func)
        
        # Continue showing field interaction
        self.wait(3)
        
        # Stop the stream animation
        stream_lines.end_animation()
        
        # Transition to duality demonstration
        self.show_physics_ml_duality(magnetic_field, self.particle_group, stream_lines)

    def show_physics_ml_duality(self, magnetic_field, particle_group, stream_lines):
        """Show the duality between physics and machine learning"""
        
        # Create catchy physics card (left side)
        physics_card = self.create_catchy_card(
            position=4 * LEFT,
            primary_color=BLUE,
            secondary_color=BLUE_D,
            title="QUANTUM"
        )
        
        # Create catchy ML card (right side)  
        ml_card = self.create_catchy_card(
            position=4 * RIGHT,
            primary_color=GREEN,
            secondary_color=GREEN_D,
            title="MACHINE LEARNING"
        )
        
        # Titles are now integrated into the cards
        
        # Create arrow connecting the cards
        connection_arrow = Arrow(
            start=physics_card.get_right() + 0.2 * RIGHT,
            end=ml_card.get_left() + 0.2 * LEFT,
            color=YELLOW,
            stroke_width=6,
            tip_length=0.3
        )
        
        # Add "DUAL" text on the arrow
        dual_text = Text("DUAL", font_size=20, color=YELLOW, weight=BOLD)
        dual_text.next_to(connection_arrow, UP, buff=0.2)
        
        
        # Filter out streamlines that are too far out
        self.filter_streamlines_for_card(stream_lines, max_distance=3.0)
        
        # Scale and move physics system to left card  
        physics_group = VGroup(magnetic_field, particle_group, stream_lines).set_z_index(10)
        physics_group.generate_target()
        physics_group.target.scale(0.4)
        physics_group.target.move_to(physics_card.get_center() + 0.5 * DOWN)
        
        self.play(
            LaggedStart(
                *[FadeIn(element) for element in physics_card],
                *[FadeIn(element) for element in ml_card],
                lag_ratio=0.1
            ),
            MoveToTarget(physics_group),
            run_time=2
        )
        
        
        # Show connection arrow
        self.play(
            GrowArrow(connection_arrow),
            Write(dual_text),
            run_time=1
        )
        
        # Add ML classifier content to right card
        self.add_ml_classifier_content(ml_card)
        

        physics_main_card = physics_card[1]  # Get the main card (has the stroke outline)
        for _ in range(3):  # Blink 3 times
            self.play(
                physics_main_card.animate.set_stroke(color=YELLOW, width=8),
                run_time=0.3
            )
            self.play(
                physics_main_card.animate.set_stroke(color=BLUE, width=4),
                run_time=0.3
            )
        
        # Then blink the right (ML) card - outline color changes  
        ml_main_card = ml_card[1]  # Get the main card (has the stroke outline)
        for _ in range(3):  # Blink 3 times
            self.play(
                ml_main_card.animate.set_stroke(color=YELLOW, width=8),
                run_time=0.3
            )
            self.play(
                ml_main_card.animate.set_stroke(color=GREEN, width=4),
                run_time=0.3
            )
        self.wait(3)
        
        all_elements = [
            physics_card, physics_group, ml_card,
            connection_arrow, dual_text,
            self.bird_pixel_grid, self.image_border, self.input_label,
            self.arrow1, self.arrow2, self.class_bars, self.class_labels
        ]

        self.play(*[FadeOut(elem) for elem in self.get_mobject_family_members()])
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            x_length=10,
            y_length=6,
            axis_config={"include_tip": False, "color": GRAY},
        )
        axes.move_to(ORIGIN)
        
        # Transform classifier box into axes
        self.play(
            Create(axes),   
            run_time=1
        )
        
        # Define 3 classes with colors
        colors = [BLUE, RED, GREEN]
        class_names = ["Class 1", "Class 2", "Class 3"]
        
        # Generate data points for each class
        np.random.seed(42)
        
        # Class 1 (bottom-left region)
        class1_points = []
        for _ in range(30):
            x = np.random.normal(-1.5, 0.8)
            y = np.random.normal(-1, 0.7)
            class1_points.append([x, y])
        
        # Class 2 (top region)
        class2_points = []
        for _ in range(30):
            x = np.random.normal(0, 0.8)
            y = np.random.normal(1.5, 0.6)
            class2_points.append([x, y])
        
        # Class 3 (bottom-right region)
        class3_points = []
        for _ in range(30):
            x = np.random.normal(1.8, 0.7)
            y = np.random.normal(-0.8, 0.8)
            class3_points.append([x, y])
        
        all_class_points = [class1_points, class2_points, class3_points]
        
        # Create dots for each class
        all_dots = VGroup()
        
        for class_points, color in zip(all_class_points, colors):
            for point in class_points:
                dot = Dot(
                    axes.c2p(point[0], point[1]),
                    color=color,
                    radius=0.08
                )
                all_dots.add(dot)
        
        # Animate dots appearing
        self.play(
            LaggedStart(
                *[FadeIn(dot, scale=0.5) for dot in all_dots],
                lag_ratio=0.02
            ),
            run_time=3
        )
        
        self.wait(1)
        
        # Draw decision boundaries (regions)
        # Create implicit regions with background fills
        
        # Region 1 (Class 1 - bottom-left)
        region1 = Polygon(
            axes.c2p(-4, -3),
            axes.c2p(0, -3),
            axes.c2p(-1, 0.5),
            axes.c2p(-4, 0.5),
            color=BLUE,
            fill_opacity=0.2,
            stroke_width=3,
            stroke_color=BLUE
        )
        
        # Region 2 (Class 2 - top)
        region2 = Polygon(
            axes.c2p(-4, 0.5),
            axes.c2p(-1, 0.5),
            axes.c2p(1.2, 0.5),
            axes.c2p(4, 1),
            axes.c2p(4, 3),
            axes.c2p(-4, 3),
            color=RED,
            fill_opacity=0.2,
            stroke_width=3,
            stroke_color=RED
        )
        
        # Region 3 (Class 3 - bottom-right)
        region3 = Polygon(
            axes.c2p(0, -3),
            axes.c2p(4, -3),
            axes.c2p(4, 1),
            axes.c2p(1.2, 0.5),
            axes.c2p(-1, 0.5),
            color=GREEN,
            fill_opacity=0.2,
            stroke_width=3,
            stroke_color=GREEN
        )
        
        regions = VGroup(region1, region2, region3)
        
        # Show regions appearing
        self.play(
            LaggedStart(
                DrawBorderThenFill(region1),
                DrawBorderThenFill(region2),
                DrawBorderThenFill(region3),
                lag_ratio=0.3
            ),
            run_time=3
        )
        
        self.wait(2)
        
        # Add labels for each region
        label1 = Text("Class 1", font_size=24, color=BLUE, weight=BOLD)
        label1.move_to(axes.c2p(-2.5, -1.5))
        
        label2 = Text("Class 2", font_size=24, color=RED, weight=BOLD)
        label2.move_to(axes.c2p(0, 2))
        
        label3 = Text("Class 3", font_size=24, color=GREEN, weight=BOLD)
        label3.move_to(axes.c2p(2.2, -1))
        
        labels = VGroup(label1, label2, label3)
        
        self.play(
            *[Write(label) for label in labels],
            run_time=1.5
        )
        
        self.wait(3)
        
        # Continue with spin-class relationship animation
        self.show_spin_class_relationship(axes, regions, labels)
    
    def show_spin_class_relationship(self, axes, regions, labels):
        """Animate the relationship between particle spin and number of classes"""
        
        # Fade out current dataset visualization
        self.play(
            *[FadeOut(obj) for obj in self.get_mobject_family_members()],
            run_time=1
        )
        
        self.wait(0.5)
        
        # First: Show Spin 1.5 in center
        spin15_particle = self.create_particle_with_label(r"\text{Spin } \frac{3}{2}", BLUE, position=ORIGIN + 2*UP)
        arrow_down_15 = Arrow(
            start=spin15_particle.get_bottom(),
            end=ORIGIN + 0.2*UP,
            color=BLUE,
            stroke_width=4,
            tip_length=0.2
        )
        
        class_boxes_4 = self.create_class_boxes(4, np.array([0, -0.7, 0]))
        
        # Animate Spin 1.5 appearing
        self.play(
            FadeIn(spin15_particle),
            run_time=1.5
        )
        
        self.play(
            GrowArrow(arrow_down_15),
            run_time=1.2
        )
        
        # Show class boxes for Spin 1.5
        self.play(
            LaggedStart(
                *[FadeIn(box, scale=0.5) for box in class_boxes_4],
                lag_ratio=0.2
            ),
            run_time=1.5
        )
        
        self.wait(1)
        
        # Move the whole Spin 1.5 group to the left
        spin15_group = VGroup(spin15_particle, arrow_down_15, class_boxes_4)
        
        self.play(
            spin15_group.animate.shift(3.5*LEFT),
            run_time=1.5
        )
        
        self.wait(0.5)
        
        # Now show Spin 2 on the right
        spin2_particle = self.create_particle_with_label(r"\text{Spin } 2", RED, position=3.5*RIGHT + 2*UP)
        arrow_down_2 = Arrow(
            start=spin2_particle.get_bottom(),
            end=3.5*RIGHT + 0.2*UP,
            color=RED,
            stroke_width=4,
            tip_length=0.2
        )
        
        class_boxes_5 = self.create_class_boxes(5, np.array([3.5*RIGHT[0], -0.7, 0]))
        
        # Animate Spin 2 appearing
        self.play(
            FadeIn(spin2_particle),
            run_time=1.5
        )
        
        self.play(
            GrowArrow(arrow_down_2),
            run_time=1.2
        )
        
        # Show class boxes for Spin 2
        self.play(
            LaggedStart(
                *[FadeIn(box, scale=0.5) for box in class_boxes_5],
                lag_ratio=0.2
            ),
            run_time=1.5
        )
        
        self.wait(2)
        
        # Show summary comparison (keep the same)
        self.show_spin_comparison()
        
        self.wait(3)
    
    def create_particle_with_label(self, spin_text, color, position):
        """Create a particle with spin label"""
        particle = Circle(radius=0.4, color=color, fill_opacity=0.8, stroke_width=4)
        glow = Circle(radius=0.6, color=color, fill_opacity=0.3, stroke_opacity=0)
        
        # Add spin label
        label = MathTex(spin_text, font_size=32, color=ORANGE)
        label.next_to(particle, UP, buff=0.3)
        
        # Add rotation arrows to indicate spin
        spin_arrow = CurvedArrow(
            start_point=particle.point_at_angle(PI/4),
            end_point=particle.point_at_angle(PI/4 + 3*PI/2),
            color=color,
            stroke_width=3
        ).scale(0.5).move_to(particle.get_center())
        
        group = VGroup(glow, particle, spin_arrow, label)
        group.move_to(position)
        
        return group
    
    def create_class_boxes(self, num_classes, center_position):
        """Create visual representation of class boxes"""
        colors = [BLUE, RED, GREEN, ORANGE, PURPLE]
        boxes = VGroup()
        
        box_width = 0.8
        total_width = num_classes * (box_width + 0.2) - 0.2
        start_x = center_position[0] - total_width / 2
        
        for i in range(num_classes):
            box = RoundedRectangle(
                width=box_width,
                height=0.8,
                corner_radius=0.1,
                color=colors[i % len(colors)],
                fill_opacity=0.6,
                stroke_width=3
            )
            
            label = Text(f"C{i+1}", font_size=20, color=WHITE, weight=BOLD)
            label.move_to(box.get_center())
            
            box_group = VGroup(box, label)
            box_group.move_to([start_x + i * (box_width + 0.2) + box_width/2, center_position[1], 0])
            boxes.add(box_group)
        
        return boxes
    
    def show_spin_comparison(self):
        """Show comparison of spin-1, spin-1.5, and spin-2 side by side"""
        
        # Create three columns
        positions = [4*LEFT, ORIGIN, 4*RIGHT]
        spins = ["Spin 1", "Spin 1.5", "Spin 2"]
        num_classes = [3, 4, 5]
        colors = [YELLOW, BLUE, RED]
        
        all_elements = VGroup()
        
        for pos, spin, num_class, color in zip(positions, spins, num_classes, colors):
            # Particle
            particle = Circle(radius=0.3, color=color, fill_opacity=0.8, stroke_width=3)
            particle.move_to(pos + 1.5*UP)
            
            # Spin label
            spin_label = Text(spin, font_size=20, color=color, weight=BOLD)
            spin_label.next_to(particle, UP, buff=0.2)
            
            # Arrow
            arrow = Arrow(
                start=particle.get_bottom(),
                end=pos + 0.5*DOWN,
                color=color,
                stroke_width=3,
                tip_length=0.2
            )
            
            # Class count
            class_text = Text(f"{num_class}\nClasses", font_size=18, color=WHITE)
            class_text.move_to(pos + 1.2*DOWN)
            
            column = VGroup(particle, spin_label, arrow, class_text)
            all_elements.add(column)
        
        # Add title
        title = Text("Spin ↔ Number of Classes", font_size=32, color=YELLOW, weight=BOLD)
        title.to_edge(DOWN).shift(UP*1.1)
        
        # Add formula showing relationship
        formula = MathTex(
            r"\text{Classes} = 2S + 1",
            font_size=40,
            color=GREEN
        )
        formula.next_to(title, DOWN, buff=0.5)
        
        self.play(
            Write(title),
            run_time=1
        )
        
        self.play(
            Write(formula),
            run_time=1
        )
        
        self.wait(3)

    def add_ml_classifier_content(self, ml_card):
        """Add machine learning classifier visualization with top-to-bottom flow"""
        
        # 1. Create pixelated bird image at the top
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bird_img_path = os.path.join(script_dir, "assets", "bird.png")
        bird_img = Image.open(bird_img_path)
        
        # Convert to RGB if it has transparency (RGBA)
        if bird_img.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', bird_img.size, (255, 255, 255))
            background.paste(bird_img, mask=bird_img.split()[-1])  # Use alpha channel as mask
            bird_img = background
        elif bird_img.mode != 'RGB':
            bird_img = bird_img.convert('RGB')
        
        # Resize to pixelated grid
        bird_pixel_size = 20  # 20x20 grid for more detail
        bird_img_resized = bird_img.resize((bird_pixel_size, bird_pixel_size), Image.Resampling.NEAREST)
        bird_img_array = np.array(bird_img_resized)
        
        # Create pixelated bird image
        bird_pixel_grid = VGroup()
        pixel_size = 0.05  # Smaller pixel size to fit more pixels
        bird_position = ml_card.get_center() + 1.0 * UP
        
        for i in range(bird_pixel_size):
            for j in range(bird_pixel_size):
                # Get RGB color from the image
                r, g, b = bird_img_array[i, j]
                
                # Convert to 0-1 range and ensure valid color
                color_rgb = np.array([r, g, b]) / 255.0
                color_rgb = np.clip(color_rgb, 0, 1)  # Ensure values are between 0 and 1
                
                # Create pixel with proper color handling
                try:
                    manim_color = rgb_to_color(color_rgb)
                except:
                    # Fallback color if conversion fails
                    manim_color = WHITE if np.mean(color_rgb) > 0.5 else BLACK
                
                pixel = Square(
                    side_length=pixel_size,
                    fill_opacity=1,
                    stroke_width=0.02,
                    stroke_color=GRAY,
                    stroke_opacity=0.5
                )
                pixel.set_fill(color=manim_color)
                
                x_pos = bird_position[0] + (j - bird_pixel_size / 2) * pixel_size
                y_pos = bird_position[1] + (bird_pixel_size / 2 - i) * pixel_size
                pixel.move_to([x_pos, y_pos, 0])
                
                bird_pixel_grid.add(pixel)
        
        # Add border around pixelated image
        image_border = SurroundingRectangle(
            bird_pixel_grid,
            color=WHITE,
            stroke_width=2,
            buff=0.05
        )
        
        # Input label
        input_label = Text("Image", font_size=12, color=WHITE)
        input_label.next_to(bird_pixel_grid, DOWN, buff=0.1)
        
        # 2. First arrow (input → classifier) - vertica
        
        # 3. Classifier box in the middle
        classifier_box = Rectangle(
            width=1.5, height=0.8,
            fill_color=GREEN_D, fill_opacity=0.8,
            stroke_color=GREEN, stroke_width=2
        )
        classifier_box.next_to(input_label, DOWN*1.5)
        classifier_label = Text("3-Class\nClassifier", font_size=10, color=WHITE)
        classifier_label.move_to(classifier_box.get_center())
        arrow1 = Arrow(
            start=input_label.get_bottom(),
            end=classifier_box.get_top() ,
            color=YELLOW,
            stroke_width=3,
            tip_length=0.15,
            buff=1
        ).scale(0.8)
        
        
        # 5. Three output classes with probability bars at the bottom
        classes = ["Dog", "Cat", "Bird"]
        class_colors = [BLUE, WHITE, RED]  # Match energy level colors
        
        class_bars = VGroup()
        class_labels = VGroup()
        
        for i, (class_name, color) in enumerate(zip(classes, class_colors)):
            # Probability bar
            bar_bg = Rectangle(width=1.8, height=0.3, color=GRAY_D, fill_opacity=0.8)
            bar_fill = Rectangle(width=0.05, height=0.3, color=color, fill_opacity=0.9)
            bar_fill.align_to(bar_bg, LEFT)
            
            bar = VGroup(bar_bg, bar_fill)
            bar.move_to(ml_card.get_center() + (1.0 - i * 0.5) * DOWN)
            
            # Class label
            label = Text(class_name, font_size=12, color=WHITE)
            label.next_to(bar, LEFT, buff=0.2)
            
            class_bars.add(bar)
            class_labels.add(label)
        
        Group(class_bars, class_labels).next_to(classifier_box, DOWN*2)
        
        arrow2 = Arrow(
            start=classifier_box.get_bottom() ,
            end=Group(class_bars, class_labels).get_top() ,
            color=YELLOW,
            stroke_width=3,
            tip_length=0.15,
            buff=1
        ).scale(0.8)
        self.play(
            LaggedStart(
                *[DrawBorderThenFill(pixel) for pixel in bird_pixel_grid],
                lag_ratio=0.01  # Faster lag for more pixels
            ),
            DrawBorderThenFill(image_border),
            Write(input_label),
            run_time=3  # Slightly longer to accommodate more pixels
        )
        
        # Step 2: Arrow to classifier
        self.play(
            GrowArrow(arrow1),
            run_time=0.8
        )
        
        # Step 3: Show classifier box
        self.play(
            DrawBorderThenFill(classifier_box),
            Write(classifier_label),
            run_time=1
        )
        
        # Step 4: Arrow to outputs
        self.play(
            GrowArrow(arrow2),
            run_time=0.8
        )
        
        # Step 5: Show output classes with bars
        self.play(
            LaggedStart(
                *[DrawBorderThenFill(bar) for bar in class_bars],
                *[Write(label) for label in class_labels],
                lag_ratio=0.2
            ),
            run_time=2
        )
        
        # Step 6: Animate probability bars growing (showing classification)
        # Simulate: 10% Dog, 15% Cat, 75% Bird
        probabilities = [0.1, 0.15, 0.75]
        
        for i, prob in enumerate(probabilities):
            bar_fill = class_bars[i][1]  # Get the fill part
            bar_fill.generate_target()
            bar_fill.target.stretch_to_fit_width(1.8 * prob)
            bar_fill.target.align_to(class_bars[i][0], LEFT)
        
        self.play(
            *[MoveToTarget(class_bars[i][1]) for i in range(3)],
            run_time=2
        )
        
        # Store all elements for transition
        self.bird_pixel_grid = bird_pixel_grid
        self.image_border = image_border
        self.input_label = input_label
        self.arrow1 = arrow1
        self.classifier_box = classifier_box
        self.classifier_label = classifier_label
        self.arrow2 = arrow2
        self.class_bars = class_bars
        self.class_labels = class_labels

    def create_catchy_card(self, position, primary_color, secondary_color, title):
        """Create a visually appealing card with gradient, shadows, and decorations"""
        
        # Main card with rounded corners effect
        main_card = RoundedRectangle(
            width=5, height=6, corner_radius=0.3,
            fill_color=BLACK, fill_opacity=0.9,
            stroke_color=primary_color, stroke_width=4
        )
        
        # Gradient effect using multiple overlays
        gradient_overlay = RoundedRectangle(
            width=4.8, height=5.8, corner_radius=0.25,
            fill_color=secondary_color, fill_opacity=0.1,
            stroke_width=0
        )
        
        # Glow effect
        glow_effect = RoundedRectangle(
            width=5.2, height=6.2, corner_radius=0.35,
            fill_color=primary_color, fill_opacity=0.2,
            stroke_width=0
        )
        

        
        # Title bar
        title_bar = RoundedRectangle(
            width=4.8, height=0.8,corner_radius=0.35,
            fill_color=primary_color, fill_opacity=0.3,
            stroke_width=0
        )
        title_bar.move_to([0, 2.4, 0])
        
        # Title text
        title_text = Text(title, font_size=16, color=primary_color, weight=BOLD)
        title_text.move_to(title_bar.get_center())
        
        # Assemble the card
        card = VGroup(
            glow_effect,
            main_card,
            gradient_overlay,
            title_bar,
            title_text
        )
        
        card.move_to(position)
        card.z_index = -15
        
        return card

    def filter_streamlines_for_card(self, stream_lines, max_distance=3.0):
        """Remove streamlines that are too far from the center"""
        # Get all the individual streamline paths
        lines_to_remove = []
        
        for i, line in enumerate(stream_lines.submobjects):
            # Check if any point in the line is too far from origin
            line_points = line.get_points()
            if len(line_points) > 0:
                # Calculate distances from origin for all points in the line
                distances = [np.linalg.norm(point[:2]) for point in line_points]
                max_line_distance = max(distances) if distances else 0
                
                # If the line extends too far, mark it for removal
                if max_line_distance > max_distance:
                    lines_to_remove.append(line)
        
        # Remove the distant streamlines
        for line in lines_to_remove:
            stream_lines.remove(line)

    def add_spin_1_particle(self, magnetic_func):
        """Add a spin-1 massive particle and show its behavior in the magnetic field"""
        
        # Create the particle (larger circle to show it's massive)
        particle = Circle(radius=0.3, color=YELLOW, fill_opacity=0.8, stroke_width=3)
        
        # Add glow effect to make it stand out
        glow = Circle(radius=0.5, color=YELLOW, fill_opacity=0.2, stroke_opacity=0)
        
        particle_group = VGroup(glow, particle)
        
        # Position particle at the center where field is strong
        center_pos = ORIGIN
        particle_group.move_to(center_pos)
        
        # Animate particle appearance
        self.play(
            FadeIn(glow, scale=0.5),
            GrowFromCenter(particle),
            run_time=2
        )
        
        # Store for later use
        self.particle_group = particle_group
        
    
        self.wait(3)
        
        # Show energy level splitting (Zeeman effect)
        energy_levels = VGroup()
        base_energy = 2 * UP + 5 * RIGHT
        
        
        
        # Final wait with all effects
        self.wait(2)


class Scene2(Scene):
    """
    Scene 2: The Quantum Meaning of Rotation
    """
    def construct(self):
        # Create main title
        title = Text(
            "The Quantum Meaning of Spin",
            font_size=52,
            color=YELLOW,
            weight=BOLD
        )
        
        
        # Create background box for emphasis
        title_bg = SurroundingRectangle(
            VGroup(title),
            color=YELLOW,
            fill_opacity=0.1,
            stroke_opacity=0.3,
            buff=0.5,
            corner_radius=0.2
        )
        
        title_group = VGroup(title_bg, title)
        
        # Animate title appearing
        self.play(
            DrawBorderThenFill(title_bg),
            run_time=1
        )
        
        self.play(
            Write(title),
            run_time=1
        )
        
        
        self.wait(2)
        
        # Move title to top of screen
        self.play(
            title_group.animate.scale(0.5).to_edge(UP, buff=0.5),
            run_time=1
        )
        
        self.wait(1)
        
        # ===== ANIMATION: Spinning particle that reveals intrinsic spin =====
        
        # Create a sphere-like particle with multiple layers for depth using Sphere
        particle = Sphere(radius=0.3, resolution=(20, 20))
        particle.set_color(YELLOW)
        particle.set_opacity(1.0)
        
        particle_mid = Sphere(radius=0.35, resolution=(15, 15))
        particle_mid.set_color(YELLOW)
        particle_mid.set_opacity(0.6)
        
        particle_glow = Sphere(radius=0.5, resolution=(15, 15))
        particle_glow.set_color(YELLOW)
        particle_glow.set_opacity(0.3)
        
        particle_group = VGroup(particle_glow, particle_mid, particle)
        particle_group.move_to(ORIGIN)
        
        # Create rotation axis along Z-axis using arrows
        axis_arrow_forward = Arrow3D(
            start=ORIGIN,
            end=2*OUT,
            color=RED,
            thickness=0.02,
            height=0.2,
            base_radius=0.05
        ).shift(OUT*10).set_z_index(-1)
        
        axis_arrow_backward = Arrow3D(
            start=ORIGIN,
            end=2*IN,
            color=RED,
            thickness=0.02,
            height=0.2,
            base_radius=0.05
        )
        
        axis_group = VGroup(axis_arrow_forward, axis_arrow_backward)
        
        # Create a curved arrow to show rotation direction around the sphere
        # Use Arc3D for a proper 3D curved arrow
        rotation_arrow = CurvedArrow(
            start_point=[0.7, 0, 0],      # starting point on the +X side
            end_point=[0, 0.7, 0],        # end point on the +Y side
            angle=-PI/2,                  # curvature of the arrow
            color=GREEN,
            stroke_width=6,
            tip_length=0.2,
        )

        # Rotate it into XY plane and bring slightly forward
        #rotation_arrow.rotate(PI / 2, axis=OUT)
        rotation_arrow.shift(0.1 * OUT).flip(Y_AXIS)
        rotation_arrow.rotate(PI / 2, axis=OUT)

        rotation_indicator = rotation_arrow       
        # Create curved motion lines around the particle
        motion_lines = VGroup()
        for angle in [0, PI/2, PI, 3*PI/2]:
            arc = Arc(
                radius=0.6,
                start_angle=angle,
                angle=PI/3,
                color=YELLOW,
                stroke_width=2,
                stroke_opacity=0.4
            )
            arc.move_to(ORIGIN)
            motion_lines.add(arc)
        
        # Show the axis first
        self.play(
            Create(axis_group),
            run_time=1
        )
        
       
        
       
        self.wait(0.3)
        
        # Show rotation direction indicator
        
        self.wait(0.3)
        
        spin_label = MathTex(r"\text{Spin} = 1", font_size=48, color=ORANGE)
        spin_label.next_to(particle_group, UP*1.5, buff=0.8)
        
        self.play(
            Write(spin_label),
            FadeIn(particle_group, scale=0.5),
            run_time=1
        )
        
        self.wait(0.5)
        self.play(
            Create(rotation_indicator),
            FadeIn(motion_lines),
            run_time=0.8
        )
        
        
        # Create a group with particle and rotation indicator to rotate together
        rotating_group = VGroup(particle_group, rotation_indicator)
        
        # Rapid spinning - rotate around the Z-axis (OUT direction)
        # This creates a 3D rotation effect around the depth axis
        self.play(
            Rotate(rotating_group, angle=4*PI, axis=OUT, about_point=ORIGIN),
            motion_lines.animate.set_stroke(opacity=0.7),
            run_time=2,
            rate_func=linear
        )
        
        # Slow down the rotation
        self.play(
            Rotate(rotating_group, angle=PI, axis=OUT, about_point=ORIGIN),
            motion_lines.animate.set_stroke(opacity=0.3),
            run_time=1.5,
            rate_func=lambda t: 1 - (1-t)**2  # Deceleration
        )
        
        # Come to a stop and fade out motion lines and rotation indicator
        self.play(
            FadeOut(motion_lines),
            FadeOut(rotation_indicator),
            FadeOut(axis_group),
            run_time=0.8
        )

        electron = Dot(color=BLUE)
        electron.set_height(1)
        electron.set_sheen(0.3, UL)
        electron.set_fill(opacity=0.8)
        # Create curved arrows for rotation visualization
        curved_arrows = VGroup(
            CurvedArrow(RIGHT*0.8, LEFT*0.8, angle=PI),
            CurvedArrow(LEFT*0.8, RIGHT*0.8, angle=PI),
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
        groups.shift(UP*0.5)

        y_ca.rotation = Rotating(
            y_ca, axis=rotate_vector(OUT, 70 * DEGREES, LEFT)
        )
        x_ca.rotation = Rotating(
            x_ca, axis=rotate_vector(OUT, 70 * DEGREES, UP)
        )
        z_ca.rotation = Rotating(z_ca, axis=OUT)
        rotations = [ca.rotation for ca in [x_ca, y_ca, z_ca]]

        # Spin-1 operator matrices
        Sx = MathTex(
            r"S_x = \frac{\hbar}{\sqrt{2}}"
            r"\begin{pmatrix}"
            r"0 & 1 & 0 \\"
            r"1 & 0 & 1 \\"
            r"0 & 1 & 0"
            r"\end{pmatrix}",
            font_size=36,
            color=ORANGE
        )

        Sy = MathTex(
            r"S_y = \frac{\hbar}{\sqrt{2}}"
            r"\begin{pmatrix}"
            r"0 & -i & 0 \\"
            r"i & 0 & -i \\"
            r"0 & i & 0"
            r"\end{pmatrix}",
            font_size=36,
            color=ORANGE
        )

        Sz = MathTex(
            r"S_z = \hbar"
            r"\begin{pmatrix}"
            r"1 & 0 & 0 \\"
            r"0 & 0 & 0 \\"
            r"0 & 0 & -1"
            r"\end{pmatrix}",
            font_size=36,
            color=ORANGE
        )

        matrices = [Sx, Sy, Sz]

        for matrix, group in zip(matrices, groups):
            matrix.next_to(group, DOWN*2)
        for matrix in matrices[1:]:
            matrix.align_to(matrices[0], DOWN)

        matrices[0].shift(LEFT*0.7)
        matrices[2].shift(RIGHT*0.5)
        self.play(ReplacementTransform(particle_group, groups))
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
        

        # Stop all rotations before highlighting
        for rotation in rotations:
            self.remove(rotation)
        
        # Highlight the Spin = 1 label
        spin_highlight = SurroundingRectangle(
            spin_label,
            color=YELLOW,
            stroke_width=4,
            buff=0.15,
            corner_radius=0.1
        )
        
        self.play(
            Create(spin_highlight),
            spin_label.animate.set_color(YELLOW),
            run_time=1
        )
        self.wait(1)
        
        # Fade out the spin label highlight
        self.play(
            FadeOut(spin_highlight),
            spin_label.animate.set_color(ORANGE),
            run_time=0.5
        )
        self.wait(0.5)
        
        # Now highlight each spin state (arrow group and matrix) one by one
        # X spin state
        x_highlight_group = SurroundingRectangle(
            x_group,
            color=GREEN,
            stroke_width=4,
            buff=0.2,
            corner_radius=0.1
        )
        x_highlight_matrix = SurroundingRectangle(
            Sx,
            color=GREEN,
            stroke_width=4,
            buff=0.15,
            corner_radius=0.1
        )
        
        self.play(
            Create(x_highlight_group),
            Create(x_highlight_matrix),
            x_group.animate.set_color(GREEN),
            Sx.animate.set_color(GREEN),
            run_time=1
        )
        self.wait(1.5)
        
        self.play(
            FadeOut(x_highlight_group),
            FadeOut(x_highlight_matrix),
            x_group.animate.set_color(WHITE),
            Sx.animate.set_color(ORANGE),
            run_time=0.5
        )
        self.wait(0.3)
        
        # Y spin state
        y_highlight_group = SurroundingRectangle(
            y_group,
            color=RED,
            stroke_width=4,
            buff=0.2,
            corner_radius=0.1
        )
        y_highlight_matrix = SurroundingRectangle(
            Sy,
            color=RED,
            stroke_width=4,
            buff=0.15,
            corner_radius=0.1
        )
        
        self.play(
            Create(y_highlight_group),
            Create(y_highlight_matrix),
            y_group.animate.set_color(RED),
            Sy.animate.set_color(RED),
            run_time=1
        )
        self.wait(1.5)
        
        self.play(
            FadeOut(y_highlight_group),
            FadeOut(y_highlight_matrix),
            y_group.animate.set_color(WHITE),
            Sy.animate.set_color(ORANGE),
            run_time=0.5
        )
        self.wait(0.3)
        
        # Z spin state
        z_highlight_group = SurroundingRectangle(
            z_group,
            color=BLUE,
            stroke_width=4,
            buff=0.2,
            corner_radius=0.1
        )
        z_highlight_matrix = SurroundingRectangle(
            Sz,
            color=BLUE,
            stroke_width=4,
            buff=0.15,
            corner_radius=0.1
        )
        
        self.play(
            Create(z_highlight_group),
            Create(z_highlight_matrix),
            z_group.animate.set_color(BLUE),
            Sz.animate.set_color(BLUE),
            run_time=1
        )
        self.wait(1.5)
        
        self.play(
            FadeOut(z_highlight_group),
            FadeOut(z_highlight_matrix),
            z_group.animate.set_color(WHITE),
            Sz.animate.set_color(ORANGE),
            run_time=0.5
        )
        
        self.wait(0.5)


class Scene3(ThreeDScene):
    """
    Scene 3: Spin as a Vector Quantity with Projections
    """
    def construct(self):
        # Set up 3D camera
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        
        # Create 3D coordinate system
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            z_length=6,
            axis_config={"color": GRAY, "stroke_width": 2}
        )
        
        # Add axis labels
        x_label = MathTex("x", font_size=36, color=RED).next_to(axes.x_axis.get_end(), RIGHT)
        y_label = MathTex("y", font_size=36, color=GREEN).next_to(axes.y_axis.get_end(), UP)
        z_label = MathTex("z", font_size=36, color=BLUE).next_to(axes.z_axis.get_end(), OUT)
        
        # Fix labels to face camera
        x_label.rotate(PI/2, axis=RIGHT)
        y_label.rotate(PI/2, axis=RIGHT)
        
        # Animate coordinate system appearing
        self.play(
            Create(axes),
            run_time=1
        )
        
        self.play(
            Write(x_label),
            Write(y_label),
            Write(z_label),
            run_time=1
        )
        
        self.wait(1)
        
        # Create spin vector (pointing at an angle in 3D space)
        spin_direction = np.array([1.5, 1.0, 2.0])
        spin_vector = Arrow3D(
            start=ORIGIN,
            end=spin_direction,
            color=YELLOW,
            thickness=0.03,
            height=0.3,
            base_radius=0.08
        )
        
        # Add vector label
        spin_label = MathTex(r"\vec{S}", font_size=48, color=YELLOW)
        spin_label.move_to(spin_direction + 0.5*RIGHT + 0.3*UP)
        spin_label.rotate(PI/2, axis=RIGHT)
        
        # Animate spin vector appearing
        self.play(
            Create(spin_vector),
            Write(spin_label),
            run_time=1
        )
        
        self.wait(1)
        
        
        
        # Now show the projections
        # S_x projection (onto x-axis)
        sx_point = np.array([spin_direction[0], 0, 0])
        sx_projection = Arrow3D(
            start=ORIGIN,
            end=sx_point,
            color=RED,
            thickness=0.02,
            height=0.2,
            base_radius=0.06
        )
        
        # Dashed line from spin tip to projection
        sx_dashed = DashedLine(
            start=spin_direction,
            end=sx_point,
            color=RED,
            dash_length=0.1,
            stroke_width=2
        )
        
        sx_label = MathTex("S_x", font_size=36, color=RED)
        sx_label.move_to(sx_point + 0.5*DOWN)
        sx_label.rotate(PI/2, axis=RIGHT)
        
        # Animate S_x projection
        self.play(
            Create(sx_dashed),
            run_time=1
        )
        self.play(
            Create(sx_projection),
            Write(sx_label),
            run_time=1
        )
        
        self.wait(1)
        
        # S_y projection (onto y-axis)
        sy_point = np.array([0, spin_direction[1], 0])
        sy_projection = Arrow3D(
            start=ORIGIN,
            end=sy_point,
            color=GREEN,
            thickness=0.02,
            height=0.2,
            base_radius=0.06
        )
        
        sy_dashed = DashedLine(
            start=spin_direction,
            end=sy_point,
            color=GREEN,
            dash_length=0.1,
            stroke_width=2
        )
        
        sy_label = MathTex("S_y", font_size=36, color=GREEN)
        sy_label.move_to(sy_point + 0.5*LEFT)
        sy_label.rotate(PI/2, axis=RIGHT)
        
        # Animate S_y projection
        self.play(
            Create(sy_dashed),
            run_time=1
        )
        self.play(
            Create(sy_projection),
            Write(sy_label),
            run_time=1
        )
        
        self.wait(1)
        
        # S_z projection (onto z-axis)
        sz_point = np.array([0, 0, spin_direction[2]])
        sz_projection = Arrow3D(
            start=ORIGIN,
            end=sz_point,
            color=BLUE,
            thickness=0.02,
            height=0.2,
            base_radius=0.06
        )
        
        sz_dashed = DashedLine(
            start=spin_direction,
            end=sz_point,
            color=BLUE,
            dash_length=0.1,
            stroke_width=2
        )
        
        sz_label = MathTex("S_z", font_size=36, color=BLUE)
        sz_label.move_to(sz_point + 0.5*RIGHT)
        sz_label.rotate(PI/2, axis=RIGHT)
        
        # Animate S_z projection
        self.play(
            Create(sz_dashed),
            run_time=1
        )
        self.play(
            Create(sz_projection),
            Write(sz_label),
            run_time=1
        )
        
        self.wait()


class Scene4(Scene):
    """
    Scene 4: Zeeman Hamiltonian Equation
    """
    def construct(self):
        equation = MathTex(
            r"H_Z = -\gamma\,\mathbf{S} \cdot \mathbf{B}",
            font_size=72,
            color=ORANGE
        )
        
        self.play(Write(equation), run_time=2)
        self.wait(3)


class Scene5(Scene):
    """
    Scene 5: Magnetic Field Equation
    """
    def construct(self):
        equation = MathTex(
            r"\mathbf{\text{B}} = B\,\hat{z}",
            font_size=72,
            color=PURE_RED
        )
        
        self.play(Write(equation), run_time=1)
        self.wait(3)





class Scene6(Scene):
    """
    Scene 6: Zeeman Hamiltonian Simplification
    """
    def construct(self):
        # Start with the full equation
        equation1 = MathTex(
            r"H_Z = -\gamma\,\mathbf{S} \cdot \mathbf{B}",
            font_size=72,
            color=ORANGE
        )
        
        self.play(Write(equation1), run_time=2)
        self.wait(2)
        
        # Transform to the simplified equation
        equation2 = MathTex(
            r"H_Z = -\gamma B S_z",
            font_size=72,
            color=ORANGE
        )
        
        self.play(ReplacementTransform(equation1, equation2), run_time=2)
        self.wait(3)



class Scene7(Scene):
    """
    Scene 7: Highlight gamma B as constant and transform to matrix
    """
    def construct(self):
        equation = MathTex(
            r"H_Z = -", r"\gamma", r" B", r" S_z",
            font_size=72,
            color=ORANGE
        )
        self.add(equation)
        brace = Brace(equation[1:3], DOWN, color=YELLOW)
        brace_text = brace.get_text("1")
        brace_text.set_color(YELLOW)
        
        self.play(
            GrowFromCenter(brace),
            Write(brace_text),
            run_time=1
        )
        self.wait(2)
        
        # Fade out brace and text
        self.play(
            FadeOut(brace),
            FadeOut(brace_text),
            run_time=0.8
        )
        
        self.wait(0.5)
        
        # Transform to matrix form
        matrix_equation = MathTex(
            r"H = \begin{pmatrix} -1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}",
            font_size=72,
            color=ORANGE
        )
        
        self.play(
            ReplacementTransform(equation, matrix_equation),
            run_time=1
        )
        diagonal_elements = Group(matrix_equation[0][4:6],matrix_equation[0][9:10],matrix_equation[0][13:14])
        circles = [Create(SurroundingRectangle(obj)) for obj in diagonal_elements]
        self.play(
            *circles,
            run_time=1
        )
        self.wait(3)


class Scene8(Scene):
    """
    Scene 8: Quantum Measurement - Rotating sphere with states
    Animate: "When we measure energy, we can only obtain diagonal values: -1, 0, or +1"
    Minimum text, maximum visuals - Quantum Roulette Style
    """
    def construct(self):
        # Create the quantum roulette wheel
        # Central circle (the wheel)
        wheel_radius = 2.5
        wheel = Circle(
            radius=wheel_radius,
            color=BLUE,
            stroke_width=8,
            fill_color=BLUE_E,
            fill_opacity=0.3
        )
        wheel.shift(0.5*LEFT)
        
        # Inner circle for aesthetics
        inner_circle = Circle(
            radius=wheel_radius * 0.7,
            color=BLUE_B,
            stroke_width=4,
            stroke_opacity=0.5
        )
        inner_circle.move_to(wheel.get_center())
        
        # Outer glow
        outer_glow = Circle(
            radius=wheel_radius * 1.1,
            color=BLUE,
            stroke_width=2,
            stroke_opacity=0.3,
            fill_opacity=0.1
        )
        outer_glow.move_to(wheel.get_center())
        
        wheel_group = VGroup(outer_glow, wheel, inner_circle)
        
        # Create 3 states positioned equally around the circle (120 degrees apart)
        states = VGroup()
        state_values = ["-1", "0", "+1"]
        state_colors = [RED, YELLOW, BLUE]
        angles = [90*DEGREES, 210*DEGREES, 330*DEGREES]  # Spread equally
        
        state_positions = []
        
        for i, (value, color, angle) in enumerate(zip(state_values, state_colors, angles)):
            # Position on the circle
            pos = wheel.get_center() + wheel_radius * np.array([np.cos(angle), np.sin(angle), 0])
            state_positions.append(pos)
            
            # Create state marker (circle with value)
            marker_circle = Circle(
                radius=0.5,
                color=color,
                fill_opacity=0.8,
                stroke_width=6,
                stroke_color=color
            )
            marker_circle.move_to(pos)
            
            # Glow effect
            marker_glow = Circle(
                radius=0.7,
                color=color,
                fill_opacity=0.3,
                stroke_width=0
            )
            marker_glow.move_to(pos)
            
            # Value label
            value_label = MathTex(value, font_size=48, color=WHITE)
            value_label.move_to(pos)
            
            state_marker = VGroup(marker_glow, marker_circle, value_label)
            states.add(state_marker)
        
        # Create the spinner arrow (starts at center, points outward)
        arrow_length = wheel_radius * 0.6
        spinner = Arrow(
            start=wheel.get_center(),
            end=wheel.get_center() + arrow_length * UP,
            color=YELLOW,
            stroke_width=8,
            tip_length=0.3,
            buff=0
        )
        
        # Add a central hub for the spinner
        hub = Circle(
            radius=0.2,
            color=YELLOW,
            fill_opacity=1,
            stroke_width=4,
            stroke_color=GOLD
        )
        hub.move_to(wheel.get_center())
        
        spinner_group = VGroup(spinner, hub)
        
        # Animate everything appearing
        self.play(
            FadeIn(outer_glow),
            DrawBorderThenFill(wheel),
            Create(inner_circle),
            run_time=1.5
        )
        
        self.play(
            LaggedStart(
                *[FadeIn(state, scale=0.5) for state in states],
                lag_ratio=0.3
            ),
            run_time=1.5
        )
        
        self.play(
            GrowArrow(spinner),
            FadeIn(hub, scale=0.5),
            run_time=1
        )
        
        self.wait(0.5)
        
        # Spinning animation - fast spin then slow down
        # First: Fast spinning (multiple rotations)
        self.play(
            Rotate(spinner_group, angle=8*PI, about_point=wheel.get_center()),
            run_time=2,
            rate_func=linear
        )
        
        
        self.wait(0.5)
        
        # Highlight the selected state (state 1 - the "0")
        selected_state = states[0]  # The "-1" state
        
        # Flash effect
        flash_circle = Circle(
            radius=0.8,
            color=RED,
            stroke_width=8,
            fill_opacity=0
        )
        flash_circle.move_to(selected_state.get_center())
        
        for _ in range(3):
            self.play(
                flash_circle.animate.scale(1.3).set_stroke(opacity=0),
                run_time=0.4
            )
            flash_circle.scale(1/1.3).set_stroke(opacity=1)
        
        self.remove(flash_circle)
        
        # Pulse the selected state
        self.play(
            selected_state.animate.scale(1.3),
            run_time=0.5
        )
        self.play(
            selected_state.animate.scale(1/1.3),
            run_time=0.5
        )
        
        self.wait(1)
        
        # Show result card appearing
        result_card = RoundedRectangle(
            width=2,
            height=1.5,
            corner_radius=0.2,
            fill_color=RED,
            fill_opacity=0.3,
            stroke_color=RED,
            stroke_width=6
        )
        result_card.next_to(wheel_group, RIGHT*4)
        
        result_value = MathTex("-1", font_size=72, color=YELLOW)
        result_value.move_to(result_card.get_center())
        
        result_label = Text("Measured", font_size=20, color=WHITE)
        result_label.next_to(result_card, UP, buff=0.3)
        
        # Animate result appearing
        self.play(
            FadeIn(result_card, scale=0.8),
            Write(result_value),
            run_time=1
        )
        
        self.play(
            Write(result_label),
            run_time=0.5
        )
        
        self.wait(1.5)


class Scene9(Scene):
    """
    Scene 9: Formula 2S + 1 with spin value boxes
    Shows the relationship between spin quantum number and number of states
    """
    def construct(self):
        # Create header formula
        header = MathTex(
            r"2S + 1",
            font_size=80,
            color=YELLOW
        )
        header.to_edge(UP, buff=0.5)
        
        # Animate header appearing
        self.play(
            Write(header),
            run_time=1.5
        )
        
        self.wait(1)
        
        # Create boxes for different spin values
        spin_values = [r"\frac{1}{2}", "1", r"\frac{3}{2}", "2"]
        colors = [BLUE, RED, GREEN, ORANGE]
        
        boxes = VGroup()
        
        for i, (spin, color) in enumerate(zip(spin_values, colors)):
            # Create box (smaller size)
            box = RoundedRectangle(
                width=1.8,
                height=1.2,
                corner_radius=0.15,
                fill_color=color,
                fill_opacity=0.2,
                stroke_color=color,
                stroke_width=3
            )
            
            # Spin label
            spin_label = MathTex(
                f"S = {spin}",
                font_size=32,
                color=color
            )
            spin_label.move_to(box.get_center())
            
            # Box group
            box_group = VGroup(box, spin_label)
            boxes.add(box_group)
        
        # Arrange boxes vertically (stacked) with smaller spacing
        boxes.arrange(DOWN, buff=0.3)
        boxes.shift(0.4*DOWN + 4*LEFT)  # Shift down and to the left
        
        # Animate boxes appearing
        self.play(
            LaggedStart(
                *[FadeIn(box, scale=0.8) for box in boxes],
                lag_ratio=0.3
            ),
            run_time=2.5
        )
        
        self.wait(1)
        
        # Function to create spinor particles for a given number of states
        def create_spinors(num_states, color, box):
            """Create spinning particles for a box with given number of states"""
            spinor_group = VGroup()
            
            for i in range(num_states):
                # Particle (electron-like dot)
                particle = Dot(color=color, radius=0.15)
                particle.set_sheen(0.3, UL)
                particle.set_fill(opacity=0.8)
                
                # Determine rotation direction based on state
                # For 3 states (S=1): first is CCW, middle is CCW (like S=2), third is CW
                # For 5 states (S=2): last spinner should be CW
                # Otherwise alternate between counter-clockwise and clockwise
                if num_states == 3:
                    if i == 0:
                        is_counter_clockwise = True
                    elif i == 1:
                        is_counter_clockwise = True  # CCW like middle of S=2
                    else:  # i == 2
                        is_counter_clockwise = False  # Clockwise
                elif num_states == 5 and i == 4:
                    is_counter_clockwise = False  # Last spinner of S=2 is CW
                else:
                    is_counter_clockwise = (i % 2 == 0)
                
                # Curved arrows for rotation visualization
                if is_counter_clockwise:
                    arc1 = Arc(radius=0.35, start_angle=0, angle=PI*0.7, color=GREY_B, stroke_width=2)
                    arc1.add_tip(tip_length=0.1)
                    arc2 = Arc(radius=0.35, start_angle=PI, angle=PI*0.7, color=GREY_B, stroke_width=2)
                    arc2.add_tip(tip_length=0.1)
                else:
                    arc1 = Arc(radius=0.35, start_angle=0, angle=-PI*0.7, color=GREY_B, stroke_width=2)
                    arc1.add_tip(tip_length=0.1)
                    arc2 = Arc(radius=0.35, start_angle=PI, angle=-PI*0.7, color=GREY_B, stroke_width=2)
                    arc2.add_tip(tip_length=0.1)
                
                curved_arrows = VGroup(arc1, arc2)
                
                # Rotation axis arrow (direction based on state index)
                # Distribute states evenly: for n states, go from up to down
                if num_states == 2:
                    arrow_direction = UP if i == 0 else DOWN
                elif num_states == 3:
                    arrow_direction = UP if i == 0 else (LEFT if i == 1 else DOWN)
                else:
                    # For more states, distribute evenly
                    angle_fraction = i / (num_states - 1)
                    arrow_direction = rotate_vector(UP, angle_fraction * PI, OUT)
                
                if np.array_equal(arrow_direction, RIGHT):
                    # Dot on top for zero spin state (pointing out of screen)
                    top_dot = Dot(color=color, radius=0.08)
                    top_dot.move_to(particle.get_center() + 0.3*OUT)
                    axis_arrow = top_dot
                else:
                    axis_arrow = Arrow(
                        ORIGIN, 0.55*arrow_direction,
                        color=color,
                        buff=0,
                        tip_length=0.15,
                        max_stroke_width_to_length_ratio=15,
                        max_tip_length_to_length_ratio=0.5
                    )
                
                # Group particle with rotation elements
                particle_group = VGroup(particle, curved_arrows, axis_arrow)
                spinor_group.add(particle_group)
            
            # Arrange spinors horizontally
            spinor_group.arrange(RIGHT, buff=0.8)
            spinor_group.next_to(box, RIGHT, buff=1.5)
            
            return spinor_group
        
        # Create all spinors for all boxes
        all_spinors = VGroup()
        num_states_list = [2, 3, 4, 5]  # S=1/2, S=1, S=3/2, S=2
        
        for i, (box, num_states) in enumerate(zip(boxes, num_states_list)):
            spinors = create_spinors(num_states, colors[i], box)
            all_spinors.add(spinors)
        
        # Animate all particles appearing at the same time
        self.play(
            LaggedStart(
                *[FadeIn(p, scale=0.5) for spinor_group in all_spinors for p in spinor_group],
                lag_ratio=0.05
            ),
            run_time=2
        )
        
        # Start continuous rotation animations for all curved arrows
        rotations = []
        for spinor_idx, spinor_group in enumerate(all_spinors):
            num_spinors = len(spinor_group)
            for i, particle_group in enumerate(spinor_group):
                particle = particle_group[0]  # The dot
                curved_arrows = particle_group[1]  # The arc arrows
                # Determine rotation direction
                # For 3 spinors (S=1): first CCW, middle CCW (like S=2), last CW
                # For 5 spinors (S=2): last spinner should be clockwise
                if num_spinors == 3:
                    if i == 0:
                        angle = 2*PI  # counter-clockwise
                    elif i == 1:
                        angle = 2*PI  # counter-clockwise (like middle of S=2)
                    else:  # i == 2
                        angle = -2*PI  # clockwise
                elif num_spinors == 5 and i == 4:
                    angle = -2*PI  # Last spinner of S=2 is clockwise
                else:
                    # For other cases, alternate
                    angle = 2*PI if i % 2 == 0 else -2*PI
                
                if angle != 0:  # Only create rotation if there's actual rotation
                    rotation = Rotating(curved_arrows, radians=angle, about_point=particle.get_center(), rate_func=linear)
                    rotations.append(rotation)
        
        # Play rotations continuously
        self.play(*rotations)
        self.play(*rotations)
        
        self.wait(15)
        


class Scene10(Scene):
    """
    Scene 10: Formula 2S + 1 with spin value boxes (static then scaled)
    Shows the relationship between spin quantum number and number of states
    """
    def construct(self):
        # Create header formula
        header = MathTex(
            r"2S + 1",
            font_size=80,
            color=YELLOW
        )
        header.to_edge(UP, buff=0.5)
        
        # Add header directly
        self.add(header)
        
        # Create boxes for different spin values
        spin_values = [r"\frac{1}{2}", "1", r"\frac{3}{2}", "2"]
        colors = [BLUE, RED, GREEN, ORANGE]
        
        boxes = VGroup()
        
        for i, (spin, color) in enumerate(zip(spin_values, colors)):
            # Create box (smaller size)
            box = RoundedRectangle(
                width=1.8,
                height=1.2,
                corner_radius=0.15,
                fill_color=color,
                fill_opacity=0.2,
                stroke_color=color,
                stroke_width=3
            )
            
            # Spin label
            spin_label = MathTex(
                f"S = {spin}",
                font_size=32,
                color=color
            )
            spin_label.move_to(box.get_center())
            
            # Box group
            box_group = VGroup(box, spin_label)
            boxes.add(box_group)
        
        # Arrange boxes vertically (stacked) with smaller spacing
        boxes.arrange(DOWN, buff=0.3)
        boxes.shift(0.4*DOWN + 4*LEFT)  # Shift down and to the left
        
        # Add boxes directly
        self.add(boxes)
        
        # Function to create spinor particles for a given number of states
        def create_spinors(num_states, color, box):
            """Create spinning particles for a box with given number of states"""
            spinor_group = VGroup()
            
            for i in range(num_states):
                # Particle (electron-like dot)
                particle = Dot(color=color, radius=0.15)
                particle.set_sheen(0.3, UL)
                particle.set_fill(opacity=0.8)
                
                # Determine rotation direction based on state
                # For 3 states (S=1): first is CCW, middle is CCW (like S=2), third is CW
                # For 5 states (S=2): last spinner should be CW
                # Otherwise alternate between counter-clockwise and clockwise
                if num_states == 3:
                    if i == 0:
                        is_counter_clockwise = True
                    elif i == 1:
                        is_counter_clockwise = True  # CCW like middle of S=2
                    else:  # i == 2
                        is_counter_clockwise = False  # Clockwise
                elif num_states == 5 and i == 4:
                    is_counter_clockwise = False  # Last spinner of S=2 is CW
                else:
                    is_counter_clockwise = (i % 2 == 0)
                
                # Curved arrows for rotation visualization
                if is_counter_clockwise:
                    arc1 = Arc(radius=0.35, start_angle=0, angle=PI*0.7, color=GREY_B, stroke_width=2)
                    arc1.add_tip(tip_length=0.1)
                    arc2 = Arc(radius=0.35, start_angle=PI, angle=PI*0.7, color=GREY_B, stroke_width=2)
                    arc2.add_tip(tip_length=0.1)
                else:
                    arc1 = Arc(radius=0.35, start_angle=0, angle=-PI*0.7, color=GREY_B, stroke_width=2)
                    arc1.add_tip(tip_length=0.1)
                    arc2 = Arc(radius=0.35, start_angle=PI, angle=-PI*0.7, color=GREY_B, stroke_width=2)
                    arc2.add_tip(tip_length=0.1)
                
                curved_arrows = VGroup(arc1, arc2)
                
                # Rotation axis arrow (direction based on state index)
                # Distribute states evenly: for n states, go from up to down
                if num_states == 2:
                    arrow_direction = UP if i == 0 else DOWN
                elif num_states == 3:
                    arrow_direction = UP if i == 0 else (LEFT if i == 1 else DOWN)
                else:
                    # For more states, distribute evenly
                    angle_fraction = i / (num_states - 1)
                    arrow_direction = rotate_vector(UP, angle_fraction * PI, OUT)
                
                if np.array_equal(arrow_direction, RIGHT):
                    # Dot on top for zero spin state (pointing out of screen)
                    top_dot = Dot(color=color, radius=0.08)
                    top_dot.move_to(particle.get_center() + 0.3*OUT)
                    axis_arrow = top_dot
                else:
                    axis_arrow = Arrow(
                        ORIGIN, 0.55*arrow_direction,
                        color=color,
                        buff=0,
                        tip_length=0.15,
                        max_stroke_width_to_length_ratio=15,
                        max_tip_length_to_length_ratio=0.5
                    )
                
                # Group particle with rotation elements
                particle_group = VGroup(particle, curved_arrows, axis_arrow)
                spinor_group.add(particle_group)
            
            # Arrange spinors horizontally
            spinor_group.arrange(RIGHT, buff=0.8)
            spinor_group.next_to(box, RIGHT, buff=1.5)
            
            return spinor_group
        
        # Create all spinors for all boxes
        all_spinors = VGroup()
        num_states_list = [2, 3, 4, 5]  # S=1/2, S=1, S=3/2, S=2
        
        for i, (box, num_states) in enumerate(zip(boxes, num_states_list)):
            spinors = create_spinors(num_states, colors[i], box)
            all_spinors.add(spinors)
        
        # Add all spinors directly without animation
        self.add(all_spinors)
        
        # Wait for 1 second
        self.wait(1)
        
        # Group boxes and spinors for scaling/shifting (excluding header)
        content = VGroup(boxes, all_spinors)
        
        # Animate scaling down and shifting left
        self.play(
            content.animate.scale(0.8).shift(1.5*LEFT),
            run_time=1
        )
        
        self.wait(2)


class Scene11(Scene):
    """
    Scene 11: Boltzmann Distribution Formula Transformation
    Shows P(E_n) formula below origin, then transforms a copy to partition function Z
    """
    def construct(self):
        # Create the first formula P(E_n) below origin
        prob_formula = MathTex(
            r"P(E_n) = \frac{e^{-\beta E_n}}{Z}",
            font_size=60,
            color=ORANGE
        )
        prob_formula.shift(1*DOWN)
        
        # Animate the probability formula appearing
        self.play(
            Write(prob_formula),
            run_time=1
        )
        
        self.wait()
        
        
        # Transform the copy into the partition function Z
        partition_formula = MathTex(
            r"Z = \sum_n e^{-\beta E_n}",
            font_size=60,
            color=YELLOW
        ).next_to(prob_formula, DOWN*2)
        
        self.play(
            ReplacementTransform(prob_formula.copy()[0][-1], partition_formula),
            run_time=1
        )
        
        
        self.wait(2)


class Scene12(Scene):
    """
    Scene 12: ML Classifier Animation (without card)
    Reproduces the multiclass classification sequence from Scene1
    """
    def construct(self):
        # 1. Create pixelated bird image at the top
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bird_img_path = os.path.join(script_dir, "assets", "bird.png")
        bird_img = Image.open(bird_img_path)
        
        # Convert to RGB if it has transparency (RGBA)
        if bird_img.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', bird_img.size, (255, 255, 255))
            background.paste(bird_img, mask=bird_img.split()[-1])  # Use alpha channel as mask
            bird_img = background
        elif bird_img.mode != 'RGB':
            bird_img = bird_img.convert('RGB')
        
        # Resize to pixelated grid
        bird_pixel_size = 25  # 20x20 grid for more detail
        bird_img_resized = bird_img.resize((bird_pixel_size, bird_pixel_size), Image.Resampling.NEAREST)
        bird_img_array = np.array(bird_img_resized)
        
        # Create pixelated bird image
        bird_pixel_grid = VGroup()
        pixel_size = 0.1  # Larger pixel size
        bird_position = 2.2 * UP  # Position at top center
        
        for i in range(bird_pixel_size):
            for j in range(bird_pixel_size):
                # Get RGB color from the image
                r, g, b = bird_img_array[i, j]
                
                # Convert to 0-1 range and ensure valid color
                color_rgb = np.array([r, g, b]) / 255.0
                color_rgb = np.clip(color_rgb, 0, 1)  # Ensure values are between 0 and 1
                
                # Create pixel with proper color handling
                try:
                    manim_color = rgb_to_color(color_rgb)
                except:
                    # Fallback color if conversion fails
                    manim_color = WHITE if np.mean(color_rgb) > 0.5 else BLACK
                
                pixel = Square(
                    side_length=pixel_size,
                    fill_opacity=1,
                    stroke_width=0.02,
                    stroke_color=GRAY,
                    stroke_opacity=0.5
                )
                pixel.set_fill(color=manim_color)
                
                x_pos = bird_position[0] + (j - bird_pixel_size / 2) * pixel_size
                y_pos = bird_position[1] + (bird_pixel_size / 2 - i) * pixel_size
                pixel.move_to([x_pos, y_pos, 0])
                
                bird_pixel_grid.add(pixel)
        
        # Add border around pixelated image
        image_border = SurroundingRectangle(
            bird_pixel_grid,
            color=WHITE,
            stroke_width=3,
            buff=0.1
        )
        
        # Input label
        input_label = Text("Image", font_size=24, color=WHITE)
        input_label.next_to(bird_pixel_grid, DOWN, buff=0.2)
        
        # 2. Classifier box in the middle
        classifier_box = Rectangle(
            width=3, height=1.5,
            fill_color=GREEN_D, fill_opacity=0.8,
            stroke_color=GREEN, stroke_width=3
        )
        classifier_box.next_to(input_label, DOWN*2)
        classifier_label = Text("3-Class\nClassifier", font_size=20, color=WHITE)
        classifier_label.move_to(classifier_box.get_center())
        
        # First arrow (input → classifier)
        arrow1 = Arrow(
            start=input_label.get_bottom(),
            end=classifier_box.get_top(),
            color=YELLOW,
            stroke_width=5,
            tip_length=0.3,
            buff=0.5
        )
        
        # 3. Three output classes with probability bars stacked horizontally at the bottom
        classes = ["Dog", "Cat", "Bird"]
        class_colors = [BLUE, WHITE, RED]  # Match energy level colors
        
        class_bars = VGroup()
        class_labels = VGroup()
        
        for i, (class_name, color) in enumerate(zip(classes, class_colors)):
            # Class label on left
            label = Text(class_name, font_size=24, color=WHITE, weight=BOLD)
            
            # Probability bar to the right of label (horizontal bar)
            bar_bg = Rectangle(width=3, height=0.6, color=GRAY_D, fill_opacity=0.8)
            bar_fill = Rectangle(width=0.05, height=0.6, color=color, fill_opacity=0.9)
            bar_fill.align_to(bar_bg, LEFT)
            
            bar = VGroup(bar_bg, bar_fill)
            bar.next_to(label, RIGHT, buff=0.3)
            
            # Group label and bar together
            class_group = VGroup(label, bar)
            
            class_bars.add(bar)
            class_labels.add(label)
        
        # Arrange all class groups horizontally (side by side)
        all_classes = VGroup()
        for i in range(3):
            all_classes.add(VGroup(class_labels[i], class_bars[i]))
        
        all_classes.arrange(RIGHT, buff=0.8)
        all_classes.next_to(classifier_box, DOWN, buff=1.3)
        
        # Second arrow (classifier → outputs)
        arrow2 = Arrow(
            start=classifier_box.get_bottom(),
            end=all_classes.get_top(),
            color=YELLOW,
            stroke_width=5,
            tip_length=0.3,
            buff=0.2
        )
        
        # === ANIMATIONS ===
        
        # Step 1: Show image
        self.play(
            LaggedStart(
                *[DrawBorderThenFill(pixel) for pixel in bird_pixel_grid],
                lag_ratio=0.01  # Faster lag for more pixels
            ),
            DrawBorderThenFill(image_border),
            Write(input_label),
            run_time=3  # Slightly longer to accommodate more pixels
        )
        
        # Step 2: Arrow to classifier
        self.play(
            GrowArrow(arrow1),
            run_time=0.8
        )
        
        # Step 3: Show classifier box
        self.play(
            DrawBorderThenFill(classifier_box),
            Write(classifier_label),
            run_time=1
        )
        
        # Step 4: Arrow to outputs
        self.play(
            GrowArrow(arrow2),
            run_time=0.8
        )
        
        # Step 5: Show output classes with bars
        self.play(
            LaggedStart(
                *[DrawBorderThenFill(bar) for bar in class_bars],
                *[Write(label) for label in class_labels],
                lag_ratio=0.2
            ),
            run_time=2
        )
        
        # Step 6: Animate probability bars growing (showing classification)
        # Simulate: 5% Dog, 5% Cat, 90% Bird
        probabilities = [0.05, 0.05, 0.90]
        
        # Create probability text labels
        prob_texts = VGroup()
        for i, prob in enumerate(probabilities):
            prob_text = Text(f"{int(prob*100)}%", font_size=28, color=YELLOW, weight=BOLD)
            prob_text.next_to(class_bars[i][0], UP, aligned_edge=RIGHT)
            prob_texts.add(prob_text)
        
        for i, prob in enumerate(probabilities):
            bar_fill = class_bars[i][1]  # Get the fill part
            bar_fill.generate_target()
            bar_fill.target.stretch_to_fit_width(3 * prob)
            bar_fill.target.align_to(class_bars[i][0], LEFT)
        
        self.play(
            *[MoveToTarget(class_bars[i][1]) for i in range(3)],
            *[FadeIn(prob_texts[i], scale=0.5) for i in range(3)],
            run_time=2
        )
        
        self.wait(2)


class Scene13(Scene):
    """
    Scene 13: Boltzmann equation for spin-1 state m = +1 (E_+1 = -1)
    """
    def construct(self):
        equation = MathTex(
            r"P(E_{+1}) = \frac{e^{\beta}}{Z}",
            font_size=60,
            color=BLUE
        )
        
        self.add(equation)
        self.wait(3)


class Scene14(Scene):
    """
    Scene 14: Boltzmann equation for spin-1 state m = 0 (E_0 = 0)
    """
    def construct(self):
        equation = MathTex(
            r"P(E_0) = \frac{1}{Z}",
            font_size=60,
            color=YELLOW
        )
        
        self.add(equation)
        self.wait(3)


class Scene15(Scene):
    """
    Scene 15: Boltzmann equation for spin-1 state m = -1 (E_-1 = +1)
    """
    def construct(self):
        equation = MathTex(
            r"P(E_{-1}) = \frac{e^{-\beta}}{Z}",
            font_size=60,
            color=RED
        )
        
        self.add(equation)
        self.wait(3)

