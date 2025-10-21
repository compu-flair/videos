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
        self.wait(2)
        
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
        spin15_particle = self.create_particle_with_label(r"\text{Spin } \frac{1}{2}", BLUE, position=ORIGIN + 2*UP)
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
        
        # Create a small glowing circle at the center
        particle = Circle(radius=0.3, color=YELLOW, fill_opacity=0.8, stroke_width=3)
        particle_glow = Circle(radius=0.5, color=YELLOW, fill_opacity=0.3, stroke_opacity=0)
        particle_group = VGroup(particle_glow, particle)
        particle_group.move_to(ORIGIN)
        
        # Add surface markers to show rotation (like a tiny planet)
        marker1 = Dot(color=WHITE, radius=0.05)
        marker2 = Dot(color=WHITE, radius=0.05)
        marker3 = Dot(color=BLUE, radius=0.06)
        
        marker1.move_to(particle.point_at_angle(PI/6))
        marker2.move_to(particle.point_at_angle(PI))
        marker3.move_to(particle.point_at_angle(-PI/3))
        
        markers = VGroup(marker1, marker2, marker3)
        
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
        
        # Particle appears with glow
        self.play(
            FadeIn(particle_group, scale=0.5),
            run_time=1
        )
        
        self.wait(0.3)
        
        # Add "Spin = 1" label
        spin_label = MathTex(r"\text{Spin} = 1", font_size=48, color=ORANGE)
        spin_label.next_to(particle_group, DOWN, buff=0.8)
        
        self.play(
            Write(spin_label),
            run_time=1
        )
        
        self.wait(0.5)
        
        # Show motion lines and markers, then spin rapidly
        self.play(
            FadeIn(motion_lines),
            FadeIn(markers),
            run_time=0.5
        )
        
        # Rapid spinning - rotate particle with markers
        spinning_group = VGroup(particle, markers)
        self.play(
            Rotate(spinning_group, angle=4*PI, about_point=ORIGIN),
            motion_lines.animate.set_stroke(opacity=0.7),
            run_time=2,
            rate_func=linear
        )
        
        # Slow down the rotation
        self.play(
            Rotate(spinning_group, angle=PI, about_point=ORIGIN),
            motion_lines.animate.set_stroke(opacity=0.3),
            run_time=1.5,
            rate_func=lambda t: 1 - (1-t)**2  # Deceleration
        )
        
        # Come to a stop and fade out markers and motion lines
        self.play(
            FadeOut(markers),
            FadeOut(motion_lines),
            run_time=0.8
        )
        
        self.wait(0.5)
        
        # Create the intrinsic spin arrow (pointing upward)
        spin_arrow = Arrow(
            start=ORIGIN,
            end=0.5*UP,
            color=GREEN,
            stroke_width=5,
            tip_length=0.25,
            buff=0
        )
        spin_arrow.move_to(particle.get_center())
        
        # Arrow fades in with slight wobble
        self.play(
            FadeIn(spin_arrow, scale=0.8),
            run_time=1
        )
        
        # Wobble the arrow slightly (quiet internal energy)
        for _ in range(3):
            self.play(
                spin_arrow.animate.rotate(angle=0.15, about_point=particle.get_center()),
                run_time=0.3
            )
            self.play(
                spin_arrow.animate.rotate(angle=-0.15, about_point=particle.get_center()),
                run_time=0.3
            )
        
        self.wait(0.5)
        
        # Particle sits there glowing steadily with pulsing glow
        self.play(
            particle_glow.animate.scale(1.3),
            rate_func=there_and_back,
            run_time=1.5
        )
        
        # Arrow makes a slow, graceful precession in place
        # Precession: the arrow rotates around the vertical axis
        self.play(
            Rotate(spin_arrow, angle=2*PI, about_point=particle.get_center()),
            particle_glow.animate.scale(1.2),
            run_time=4,
            rate_func=smooth
        )
        
        # Return glow to normal
        self.play(
            particle_glow.animate.scale(1/1.2),
            run_time=0.5
        )
        
        self.wait(2)
        
        # Final fade out
        self.play(
            *[FadeOut(mob) for mob in [particle_group, spin_arrow, spin_label]],
            run_time=1.5
        )




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
