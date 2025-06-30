from manim_imports_ext import *


class MiniLessonTitle(InteractiveScene):
    def construct(self):
        title = Text("Mini-lesson on complex exponents", font_size=72)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()


class ConfusionAndWhy(TeacherStudentsScene):
    def construct(self):
        # Test
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(200)
        morty = self.teacher
        stds = self.students

        q_marks = Tex(R"???")
        q_marks.space_out_submobjects(1.5)
        q_marks.next_to(stds[0], UP, MED_SMALL_BUFF)
        self.play(
            self.change_students("confused", "pondering", "pleading", look_at=self.screen),
            FadeIn(q_marks, UP, lag_ratio=0.5),
            morty.change("raise_right_hand")
        )
        self.wait(3)
        self.play(morty.change("raise_left_hand", look_at=3 * UR))
        self.play(
            self.change_students("erm", "thinking", "hesitant", look_at=morty.get_top() + 2 * UP),
            FadeOut(q_marks)
        )
        self.wait(4)
        self.play(self.change_students("pondering"))
        self.wait(3)


class DerivativeOfExp(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        tex_kw = dict(t2c={"t": GREY_B, "s": YELLOW})
        equation = Tex(R"\frac{d}{dt} e^{st} = s \cdot e^{st}", font_size=90, **tex_kw)
        deriv_part = equation[R"\frac{d}{dt}"][0]
        exp_parts = equation[R"e^{st}"]
        equals = equation[R"="][0]
        s_dot = equation[R"s \cdot"][0]

        v_box = SurroundingRectangle(VGroup(deriv_part, exp_parts[0]))
        p_box = SurroundingRectangle(exp_parts[1])
        s_box = SurroundingRectangle(s_dot)
        s_box.match_height(p_box, stretch=True).match_y(p_box)
        boxes = VGroup(v_box, p_box, s_box)
        boxes.set_stroke(width=2)
        boxes.set_submobject_colors_by_gradient(GREEN, BLUE, YELLOW)

        v_label = Text("Velocity", font_size=48).match_color(v_box)
        p_label = Text("Position", font_size=48).match_color(p_box)
        s_label = Text("Modifier", font_size=48).match_color(s_box)
        v_label.next_to(v_box, UP, MED_SMALL_BUFF)
        p_label.next_to(p_box, UP, MED_SMALL_BUFF, aligned_edge=LEFT)
        s_label.next_to(s_box, DOWN, MED_SMALL_BUFF)
        labels = VGroup(v_label, p_label, s_label)

        frame.move_to(exp_parts[0])

        self.add(exp_parts[0])
        self.wait()
        self.play(Write(deriv_part))
        self.play(
            TransformFromCopy(*exp_parts, path_arc=90 * DEG),
            Write(equals),
            frame.animate.center(),
        )
        self.play(
            TransformFromCopy(exp_parts[1][1], s_dot[0], path_arc=90 * DEG),
            Write(s_dot[1]),
        )
        self.wait()

        # Show labels
        for box, label in zip(boxes, labels):
            self.play(ShowCreation(box), FadeIn(label))

        self.wait()
        full_group = VGroup(equation, boxes, labels)

        # Set s equal to 1
        s_eq_1 = Tex(R"s = 1", font_size=72, **tex_kw)
        simple_equation = Tex(R"\frac{d}{dt} e^{t} = e^{t}", font_size=72, **tex_kw)
        simple_equation.to_edge(UP).shift(2 * LEFT)
        s_eq_1.next_to(simple_equation, RIGHT, buff=2.5)
        arrow = Arrow(s_eq_1, simple_equation, thickness=5, buff=0.35).shift(0.05 * DOWN)

        self.play(
            Write(s_eq_1),
            GrowArrow(arrow),
            TransformMatchingTex(equation.copy(), simple_equation, run_time=1.5, lag_ratio=0.02),
            full_group.animate.shift(DOWN).scale(0.75).fade(0.15)
        )
        self.wait()


class HighlightRect(InteractiveScene):
    def construct(self):
        img = ImageMobject('/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2025/laplace/exponentials/DynamicExpIntuitionStill.png')
        img.set_height(FRAME_HEIGHT)

        # Rects
        rects = VGroup(
            Rectangle(2.25, 1).move_to((2.18, 2.74, 0)),
            Rectangle(3, 2.5).move_to((-5.52, -1.73, 0)),
        )
        rects.set_stroke(YELLOW, 2)

        self.play(ShowCreation(rects[0]))
        self.play(TransformFromCopy(*rects))
        self.play(FadeOut(rects))


class DefineI(InteractiveScene):
    def construct(self):
        eq = Tex(R"i = \sqrt{-1}", t2c={"i": YELLOW}, font_size=90)
        self.play(Write(eq))
        self.wait()


class WaitWhy(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.students[0].change("erm", self.screen),
            self.students[1].change("tease", self.screen),
            self.students[2].says("Wait, why?", "confused", look_at=self.screen, bubble_direction=LEFT),
        )
        self.wait(4)


class MultiplicationByI(InteractiveScene):
    def construct(self):
        # Example number
        plane = ComplexPlane(
            background_line_style=dict(stroke_color=BLUE, stroke_width=1),
            # faded_line_style=dict(stroke_color=BLUE, stroke_width=0.5, stroke_opacity=0.5),
        )
        plane.add_coordinate_labels(font_size=24)

        z = 3 + 2j
        tex_kw = dict(t2c={"a": YELLOW, "b": PINK})

        vect = Vector(plane.n2p(z), fill_color=WHITE, thickness=4)
        vect_label = Tex(R"a + bi", **tex_kw)
        vect_label.next_to(vect.get_end(), UR, SMALL_BUFF)
        vect_label.set_backstroke(BLACK, 5)

        lines = VGroup(
            Line(ORIGIN, plane.n2p(z.real)).set_color(YELLOW),
            Line(plane.n2p(z.real), plane.n2p(z)).set_color(PINK),
        )
        a_label, b_label = line_labels = VGroup(
            Tex(R"a", font_size=36, **tex_kw).next_to(lines[0], UP, SMALL_BUFF),
            Tex(R"bi", font_size=36, **tex_kw).next_to(lines[1], RIGHT, SMALL_BUFF),
        )
        line_labels.set_backstroke(BLACK, 5)

        self.add(plane, Point(), plane.coordinate_labels)
        self.add(vect)
        self.add(vect_label)
        for line, label in zip(lines, line_labels):
            self.play(
                ShowCreation(line),
                FadeIn(label, 0.25 * line.get_vector())
            )
        self.wait()

        # Multiply components by i
        new_lines = lines.copy()
        new_lines.rotate(90 * DEG, about_point=ORIGIN)
        new_lines[1].move_to(ORIGIN, RIGHT)

        new_a_label = Tex(R"ai", font_size=36, **tex_kw).next_to(new_lines[0], RIGHT, SMALL_BUFF)
        new_b_label = Tex(R"bi \cdot i", font_size=36, **tex_kw).next_to(new_lines[1], UP, SMALL_BUFF)
        neg_b_label = Tex(R"=-b", font_size=36, **tex_kw)
        neg_b_label.move_to(new_b_label.get_right())

        mult_i_label = Tex(R"\times i", font_size=90)
        mult_i_label.set_backstroke(BLACK, 5)
        mult_i_label.to_corner(UR, buff=MED_LARGE_BUFF).shift(0.2 * UP)

        self.play(Write(mult_i_label))
        self.wait()
        self.play(
            TransformFromCopy(lines[0], new_lines[0], path_arc=90 * DEG),
            TransformFromCopy(a_label[0], new_a_label[0], path_arc=90 * DEG),
            TransformFromCopy(mult_i_label[1], new_a_label[1]),
        )
        self.wait()
        self.play(
            TransformFromCopy(lines[1], new_lines[1], path_arc=90 * DEG),
            TransformFromCopy(b_label[0], new_b_label[:-1], path_arc=90 * DEG),
            TransformFromCopy(mult_i_label[1], new_b_label[-1]),
        )
        self.wait()
        self.play(
            FlashAround(VGroup(new_b_label, new_lines[1]), color=PINK, time_width=1.5, run_time=2),
            new_b_label.animate.next_to(neg_b_label, LEFT, SMALL_BUFF),
            FadeIn(neg_b_label, SMALL_BUFF * RIGHT),
        )
        self.wait()
        self.play(VGroup(new_lines[1], new_b_label, neg_b_label).animate.shift(new_lines[0].get_vector()))

        # New vector
        vect_copy = vect.copy()
        elbow = Elbow().rotate(vect.get_angle(), about_point=ORIGIN)
        self.play(
            Rotate(vect_copy, 90 * DEG, run_time=2, about_point=ORIGIN),
        )
        self.play(
            ShowCreation(elbow)
        )
        self.wait()

    def old_material(self):
        # Show the algebra
        algebra = VGroup(
            Tex(R"i \cdot (a + bi)", **tex_kw),
            Tex(R"ai + bi^2", **tex_kw),
            Tex(R"-b + ai", **tex_kw),
        )
        algebra.set_backstroke(BLACK, 8)
        algebra.arrange(DOWN, buff=0.35)
        algebra.to_corner(UL)

        self.play(
            TransformFromCopy(vect_label, algebra[0]["a + bi"][0]),
            FadeIn(algebra[0]),
        )
        self.play(LaggedStart(
            TransformFromCopy(algebra[0]["a"], algebra[1]["a"]),
            TransformFromCopy(algebra[0]["+ bi"], algebra[1]["+ bi"]),
            TransformFromCopy(algebra[0]["i"][0], algebra[1]["i"][0]),
            TransformFromCopy(algebra[0]["i"][0], algebra[1]["2"]),
            lag_ratio=0.25
        ))
        self.wait()
        self.play(LaggedStart(
            TransformFromCopy(algebra[1]["bi^2"], algebra[2]["-b"]),
            TransformFromCopy(algebra[1]["ai"], algebra[2]["ai"]),
            TransformFromCopy(algebra[1]["+"], algebra[2]["+"]),
            lag_ratio=0.25
        ))
        self.wait()

        # New lines
        new_lines = lines.copy()
        new_lines.rotate(90 * DEG)
        new_lines.refresh_bounding_box()
        new_lines[1].move_to(ORIGIN, RIGHT)
        new_lines[0].move_to(new_lines[1].get_left(), DOWN)

        neg_b_label = Tex(R"-b", fill_color=PINK, font_size=36).next_to(new_lines[1], UP, SMALL_BUFF)
        new_a_label = Tex(R"a", fill_color=YELLOW, font_size=36).next_to(new_lines[0], LEFT, SMALL_BUFF)

        self.play(
            TransformFromCopy(lines[1], new_lines[1]),
            FadeTransform(algebra[2]["-b"].copy(), neg_b_label),
        )
        self.play(
            TransformFromCopy(lines[0], new_lines[0]),
            FadeTransform(algebra[2]["a"].copy(), new_a_label),
        )
        self.wait()


class UnitArcLengthsOnCircle(InteractiveScene):
    def construct(self):
        # Moving sectors
        arc = Arc(0, 1, radius=2.5, stroke_color=GREEN, stroke_width=8)
        sector = Sector(angle=1, radius=2.5).set_fill(GREEN, 0.25)
        v_line = Line(ORIGIN, 2.5 * UP)
        v_line.match_style(arc)
        v_line.move_to(arc.get_start(), DOWN)

        self.add(v_line)
        self.play(
            FadeIn(sector),
            ReplacementTransform(v_line, arc),
        )

        group = VGroup(sector, arc)
        self.add(group)

        for n in range(5):
            self.wait(2)
            group.rotate(1, about_point=ORIGIN)

        return

        # Previous
        colors = [RED, BLUE]
        arcs = VGroup(
            Arc(n, 1, radius=2.5, stroke_color=colors[n % 2], stroke_width=8)
            for n in range(6)
        )
        for arc in arcs:
            one = Integer(1, font_size=24).move_to(1.0 * arc.get_center())
            self.play(ShowCreation(arc, rate_func=linear, run_time=2))
        self.wait()


class SimpleIndicationRect(InteractiveScene):
    def construct(self):
        rect = Rectangle(3, 2)
        # Test
        self.play(FlashAround(rect, time_width=2.0, run_time=2, color=WHITE))


class WriteSPlane(InteractiveScene):
    def construct(self):
        title = Text("S-plane", font_size=72)
        title.set_color(YELLOW)
        self.play(Write(title))
        self.wait()


class WriteMu(InteractiveScene):
    def construct(self):
        sym = Tex(R"\mu")
        rect = SurroundingRectangle(sym, buff=0.05)
        rect.set_stroke(YELLOW, 2)
        mu = TexText("``Mu''")
        mu.set_color(YELLOW)
        mu.next_to(rect, DOWN)
        self.play(
            Write(mu),
            ShowCreation(rect)
        )
        self.wait()
