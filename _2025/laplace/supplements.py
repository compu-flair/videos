from manim_imports_ext import *


class WhoCares(TeacherStudentsScene):
    def construct(self):
        # Test
        self.remove(self.background)
        stds = self.students
        morty = self.teacher

        self.play(
            stds[2].says("Who cares?", mode="angry", look_at=3 * UP),
            morty.change("guilty", stds[2].eyes),
            stds[1].change("hesitant", 3 * UP),
            stds[0].change("erm", stds[2].eyes),
        )
        self.wait(3)


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


class WhatAndWhy(InteractiveScene):
    def construct(self):
        words = VGroup(
            Tex(R"\text{1) Understanding } e^{i {t}} \\ \text{ intuitively}", t2c={R"{t}": GREY_B}),
            TexText(R"2) How they \\ naturally arise"),
        )
        words[0][R"intuitively"].align_to(words[0]["Understanding"], LEFT)
        words[1][R"naturally arise"].align_to(words[1]["How"], LEFT)
        words.scale(1.25)
        for word, u in zip(words, [1, -1]):
            word.next_to(ORIGIN, RIGHT)
            word.set_y(u * FRAME_HEIGHT / 4)
            self.play(Write(word))
            self.wait()
        # Test


class OtherExponentialDerivatives(InteractiveScene):
    def construct(self):
        # Test
        kw = dict(t2c={"t": GREY_B})
        derivs = VGroup(
            Tex(R"\frac{d}{dt} 2^t = (0.693...)2^t", **kw),
            Tex(R"\frac{d}{dt} 3^t = (1.098...)3^t", **kw),
            Tex(R"\frac{d}{dt} 4^t = (1.386...)4^t", **kw),
            Tex(R"\frac{d}{dt} 5^t = (1.609...)5^t", **kw),
            Tex(R"\frac{d}{dt} 6^t = (1.791...)6^t", **kw),
        )
        derivs.scale(0.75)
        derivs.arrange(DOWN, buff=0.7)
        derivs.to_corner(UL)

        self.play(LaggedStartMap(FadeIn, derivs, shift=UP, lag_ratio=0.5, run_time=5))
        self.wait()


class VariousExponentials(InteractiveScene):
    def construct(self):
        # Test
        exp_st = Tex(R"e^{st}", t2c={"s": YELLOW, "t": BLUE}, font_size=90)
        gen_exp = Tex(R"e^{+0.50 t}", t2c={"+0.50": YELLOW, "t": BLUE}, font_size=90)
        exp_st.to_edge(UP, buff=MED_LARGE_BUFF)
        gen_exp.move_to(exp_st)

        num = gen_exp["+0.50"]
        num.set_opacity(0)
        gen_exp["t"].scale(1.25, about_edge=UL)

        s_num = DecimalNumber(-1.00, edge_to_fix=ORIGIN, include_sign=True)
        s_num.set_color(YELLOW)
        s_num.replace(num, dim_to_match=1)

        self.add(gen_exp, s_num)
        self.play(ChangeDecimalToValue(s_num, 0.5, run_time=4))
        self.wait()
        self.play(LaggedStart(
            ReplacementTransform(gen_exp["e"][0], exp_st["e"][0]),
            ReplacementTransform(s_num, exp_st["s"]),
            ReplacementTransform(gen_exp["t"][0], exp_st["t"][0]),
        ))
        self.wait()


class WhyToWhat(InteractiveScene):
    def construct(self):
        # Title text
        why = Text("Why", font_size=90)
        what = Text("Wait, what does this even mean?", font_size=72)
        VGroup(why, what).to_edge(UP)

        what_word = what["what"][0].copy()
        what["what"][0].set_opacity(0)

        arrow = Arrow(
            what["this"].get_bottom(),
            (2.5, 2, 0),
            thickness=5,
            fill_color=YELLOW
        )

        self.play(FadeIn(why, UP))
        self.wait()
        self.play(
            # FadeOut(why, UP),
            ReplacementTransform(why, what_word),
            FadeIn(what, lag_ratio=0.1),
        )
        self.play(
            GrowArrow(arrow),
            what["this"].animate.set_color(YELLOW)
        )
        self.wait()


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
        self.add(img)

        # Rects
        rects = VGroup(
            Rectangle(2.25, 1).move_to((2.18, 2.74, 0)),
            Rectangle(2, 0.85).move_to((-5.88, -2.2, 0.0)),
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


class ODEStoExp(InteractiveScene):
    def construct(self):
        # Test
        odes, exp = words = VGroup(
            Text("Differential\nEquations"),
            Tex("e^{st}", t2c={"s": YELLOW}, font_size=72),
        )
        exp.match_height(odes)
        words.arrange(RIGHT, buff=3.0)
        words.to_edge(UP, buff=1.25)

        top_arrow, low_arrow = arrows = VGroup(
            Arrow(odes.get_corner(UR), exp.get_corner(UL), path_arc=-60 * DEG, thickness=5),
            Arrow(exp.get_corner(DL), odes.get_corner(DR), path_arc=-60 * DEG, thickness=5),
        )
        arrows.set_fill(TEAL)

        top_words = Tex(R"Explain", font_size=36).next_to(top_arrow, UP, SMALL_BUFF)
        low_words = Tex(R"Solves", font_size=36).next_to(low_arrow, DOWN, SMALL_BUFF)

        exp.shift(0.25 * UP + 0.05 * LEFT)

        self.add(words)
        self.wait()
        self.play(
            Write(top_arrow),
            Write(top_words),
        )
        self.wait()
        self.play(
            # Write(low_arrow),
            TransformFromCopy(top_arrow, low_arrow, path_arc=-PI),
            Write(low_words),
        )
        self.wait()


class VLineOverZero(InteractiveScene):
    def construct(self):
        # Test
        rect = Square(0.25)
        rect.move_to(2.5 * DOWN)
        v_line = Line(rect.get_top(), 4 * UP, buff=0.1)
        v_line.set_stroke(YELLOW, 2)
        rect.match_style(v_line)

        self.play(
            ShowCreationThenFadeOut(rect),
            ShowCreationThenFadeOut(v_line),
        )
        self.wait()


class KIsSomeConstant(InteractiveScene):
    def construct(self):
        rect = SurroundingRectangle(Text("k"), buff=0.05)
        rect.set_stroke(YELLOW, 2)
        words = Text("Some constant", font_size=24)
        words.next_to(rect, UP, SMALL_BUFF)
        words.match_color(rect)

        self.play(ShowCreation(rect), FadeIn(words))
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


class ReferenceGuessingExp(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.remove(self.background)
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)

        # Student asks
        question = Tex(R"x(t) = ???")
        lhs = question["x(t)"][0]
        rhs = question["= ???"][0]
        bubble = stds[2].get_bubble(question, bubble_type=SpeechBubble, direction=LEFT)
        lhs.save_state()
        lhs.scale(0.25).move_to([-6.24, 2.38, 0])

        self.play(
            morty.change("hesitant", look_at=stds[2].eyes),
            self.change_students("erm", "confused", "maybe", look_at=self.screen)
        )
        self.wait()
        self.play(
            stds[2].change("raise_left_hand", morty.eyes),
            Write(bubble[0]),
            Write(rhs, time_span=(0.5, 1.0)),
            Restore(lhs),
        )
        self.wait()
        self.add(Point())
        self.play(
            morty.says("Here's a trick:", mode="tease", bubble_creation_class=FadeIn),
            self.change_students("pondering", "thinking", "hesitant", look_at=UL),
        )
        self.wait(2)

        # Teacher gestures to upper right, students look confused and hesitant
        eq_point = 5 * RIGHT + 3 * UP
        self.play(
            morty.change("raise_right_hand", look_at=eq_point),
            FadeOut(bubble),
            FadeOut(morty.bubble),
            self.change_students("confused", "thinking", "hesitant", look_at=eq_point),
        )
        self.wait()
        self.play(self.change_students("confused", "hesitant", "confused", look_at=eq_point, lag_ratio=0.1))
        self.wait()
        self.play(
            morty.change("shruggie", look_at=eq_point),
        )
        self.wait(2)
        self.play(
            self.change_students("angry", "hesitant", "erm", look_at=morty.eyes),
            morty.animate.look_at(stds)
        )
        self.wait(2)

        # Transition: flip and reposition morty to where stds are
        new_teacher_pos = stds[2].get_bottom()
        new_teacher = morty.copy()
        new_teacher.change_mode("raise_left_hand")
        new_teacher.look_at(3 * UR)
        new_teacher.body.set_color(GREY_C)

        self.play(
            morty.animate.scale(0.8).flip().change_mode("confused").look_at(5 * UR).move_to(new_teacher_pos, DOWN),
            LaggedStartMap(FadeOut, stds, shift=DOWN, lag_ratio=0.2, run_time=1),
            FadeIn(new_teacher, time_span=(0.5, 1.5)),
        )
        self.play(morty.change("pleading", 3 * UR))
        self.play(Blink(new_teacher))
        self.wait(2)
        self.play(LaggedStart(
            morty.change("erm", new_teacher.eyes),
            new_teacher.change("guilty", look_at=morty.eyes),
            lag_ratio=0.5,
        ))
        self.wait(3)

        # Reference a graph
        self.play(
            morty.change("angry", 2 * UR),
            new_teacher.change("tease", 2 * UR)
        )
        self.play(Blink(morty))
        self.play(Blink(new_teacher))
        self.wait()


class JustAlgebra(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer(mode="tease")
        morty.body.insert_n_curves(100)
        self.play(morty.says("Just algebra!", mode="hooray", look_at=2 * UL))
        self.play(Blink(morty))
        self.wait()
        self.play(
            FadeOut(morty.bubble),
            morty.change("tease", look_at=2 * UL + UP)
        )
        self.play(Blink(morty))
        self.wait()


class BothPositiveNumbers(InteractiveScene):
    def construct(self):
        tex = Tex("k / m")
        self.add(tex)

        # Test
        rects = VGroup(SurroundingRectangle(tex[c], buff=0.05) for c in "km")
        rects.set_stroke(GREEN, 3)
        plusses = VGroup(Tex(R"+").next_to(rect, DOWN, SMALL_BUFF) for rect in rects)
        plusses.set_fill(GREEN)

        self.play(
            LaggedStartMap(ShowCreation, rects, lag_ratio=0.5),
            LaggedStartMap(FadeIn, plusses, shift=0.25 * DOWN, lag_ratio=0.5)
        )
        self.wait()


class ButSpringsAreReal(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        self.play(
            stds[0].change("maybe", self.screen),
            stds[1].says("But...springs are real", mode="confused", look_at=self.screen),
            stds[2].change("erm", self.screen),
            morty.change("tease", stds[2].eyes)
        )
        self.wait(4)


class ShowIncreaseToK(InteractiveScene):
    def construct(self):
        # Test
        k = Tex(R"k")

        box = SurroundingRectangle(k)
        box.set_stroke(GREEN, 5)
        arrow = Vector(UP, thickness=6)
        arrow.set_fill(GREEN)
        center = box.get_center()

        self.play(
            ShowCreation(box),
            UpdateFromAlphaFunc(
                arrow, lambda m, a: m.move_to(
                    center + interpolate(-1, 1, a) * UP
                ).set_fill(
                    opacity=there_and_back(a) * 0.7
                ),
                run_time=4
            ),
        )
        self.wait()


class LinearityDefinition(InteractiveScene):
    def construct(self):
        # Base differential equation string
        eq_str = R"m x''(t) + k x(t) = 0"
        t2c = {"x_1": TEAL, "x_2": RED, "0.0": YELLOW, "2.0": YELLOW}

        base_eq = Tex(eq_str)
        base_eq.to_edge(UP)

        eq1, eq2, eq3, eq4 = equations = VGroup(
            Tex(eq_str.replace("x", "x_1"), t2c=t2c),
            Tex(eq_str.replace("x", "x_2"), t2c=t2c),
            Tex(R"m\Big(x_1''(t) + x_2''(t) \Big) + k \Big(x_1(t) + x_2(t)\Big) = 0", t2c=t2c),
            Tex(R"m\Big(0.0 x_1''(t) + 2.0 x_2''(t) \Big) + k \Big(0.0 x_1(t) + 2.0 x_2(t)\Big) = 0", t2c=t2c),
        )
        for eq in equations:
            eq.set_max_width(7)
        equations.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        equations.to_edge(RIGHT)
        equations.shift(DOWN)

        phrase1, phrase2, phrase3, phrase4 = phrases = VGroup(
            TexText("If $x_1$ solves it:", t2c=t2c),
            TexText("and $x_2$ solves it:", t2c=t2c),
            TexText("Then $(x_1 + x_2)$ solves it:", t2c=t2c),
            TexText("Then $(0.0 x_1 + 2.0 x_2)$ solves it:", t2c=t2c),
        )

        for phrase, eq in zip(phrases, equations):
            phrase.set_max_width(5)
            phrase.next_to(eq, LEFT, LARGE_BUFF)

        eq4.move_to(eq3)
        phrase4.move_to(phrase3)

        kw = dict(edge_to_fix=RIGHT)
        c1_terms = VGroup(phrase4.make_number_changeable("0.0", **kw), *eq4.make_number_changeable("0.0", replace_all=True, **kw))
        c2_terms = VGroup(phrase4.make_number_changeable("2.0", **kw), *eq4.make_number_changeable("2.0", replace_all=True, **kw))

        # Show base equation
        self.play(Write(phrase1), FadeIn(eq1))
        self.wait()
        self.play(
            TransformMatchingTex(eq1.copy(), eq2, key_map={"x_1": "x_2"}, run_time=1, lag_ratio=0.01),
            FadeTransform(phrase1.copy(), phrase2)
        )
        self.wait()
        self.play(
            FadeIn(phrase3, DOWN),
            FadeIn(eq3, DOWN),
        )
        self.wait()
        self.play(
            FadeOut(eq3, 0.5 * DOWN),
            FadeOut(phrase3, 0.5 * DOWN),
            FadeIn(eq4, 0.5 * DOWN),
            FadeIn(phrase4, 0.5 * DOWN),
        )
        for _ in range(8):
            new_c1 = random.random() * 10
            new_c2 = random.random() * 10
            self.play(*(
                ChangeDecimalToValue(c1, new_c1, run_time=1)
                for c1 in c1_terms
            ))
            self.wait(0.5)
            self.play(*(
                ChangeDecimalToValue(c2, new_c2, run_time=1)
                for c2 in c2_terms
            ))
            self.wait(0.5)


class GeneralLinearEquation(InteractiveScene):
    def construct(self):
        # Set up equations
        a_texs = ["a_n", "a_2", "a_1", "a_0"]
        x_texs = ["x^{n}(t)", "x''(t)", "x'(t)", "x(t)"]
        x_colors = color_gradient([BLUE, TEAL], len(x_texs), interp_by_hsl=True)
        t2c = {"{s}": YELLOW}
        t2c.update({a: WHITE for a in a_texs})
        t2c.update({x: color for x, color in zip(x_texs, x_colors)})
        ode = Tex(R"a_n x^{n}(t) + \cdots + a_2 x''(t) + a_1 x'(t) + a_0 x(t) = 0", t2c=t2c)
        exp_version = Tex(
            R"a_n \left({s}^n e^{{s}t}\right) "
            R"+ \cdots "
            R"+ a_2 \left({s}^2 e^{{s}t}\right) "
            R"+ a_1 \left({s}e^{{s}t}\right) "
            R"+ a_0 e^{{s}t} = 0",
            t2c=t2c
        )
        factored = Tex(R"e^{{s}t} \left(a_n {s}^n + \cdots + a_2 {s}^2 + a_1 {s} + a_0 \right) = 0", t2c=t2c)

        ode.to_edge(UP)
        exp_version.next_to(ode, DOWN, MED_LARGE_BUFF)
        factored.move_to(exp_version)

        # Introduce ode
        x_arrows = VGroup(
            Arrow(UP, ode[x_tex].get_bottom(), fill_color=color)
            for x_tex, color in zip(x_texs, x_colors)
        )
        x_arrows.reverse_submobjects()
        index = ode.submobjects.index(ode["a_2"][0][0])

        right_part = ode[index:]
        left_part = ode[:index]
        right_part.save_state()
        right_part.set_x(0)

        self.play(FadeIn(right_part, UP))
        self.wait()
        self.play(LaggedStart(
            Restore(right_part),
            Write(left_part)
        ))
        self.add(ode)
        self.play(LaggedStartMap(VFadeInThenOut, x_arrows, lag_ratio=0.1, run_time=3))
        self.wait()

        # Plug in e^{st}
        key_map = {
            R"+ a_0 x(t) = 0": R"+ a_0 e^{{s}t} = 0",
            R"+ a_1 x'(t)": R"+ a_1 \left({s}e^{{s}t}\right)",
            R"+ a_2 x''(t)": R"+ a_2 \left({s}^2 e^{{s}t}\right)",
            R"+ \cdots": R"+ \cdots",
            R"a_n x^{n}(t)": R"a_n \left({s}^n e^{{s}t}\right)",
        }

        self.play(LaggedStart(*(
            FadeTransform(ode[k1].copy(), exp_version[k2])
            for k1, k2 in key_map.items()
        ), lag_ratio=0.6, run_time=4))
        self.wait()
        self.play(
            TransformMatchingTex(
                exp_version,
                factored,
                matched_keys=[R"e^{{s}t}", "{s}^n", "{s}^2", "{s}", "a_n", "a_2", "a_1", "a_0"],
                path_arc=45 * DEG
            )
        )
        self.wait()

        # Highlight the polynomail
        poly_rect = SurroundingRectangle(factored[R"\left(a_n {s}^n + \cdots + a_2 {s}^2 + a_1 {s} + a_0 \right)"])
        poly_rect.set_stroke(YELLOW, 1)

        self.play(ShowCreation(poly_rect))

        # Plane
        plane = ComplexPlane((-3, 3), (-3, 3), width=6, height=6)
        plane.set_height(4.5)
        plane.next_to(poly_rect, DOWN, LARGE_BUFF)
        plane.set_x(0)
        plane.add_coordinate_labels(font_size=16)

        roots = [0.2 + 1j, 0.2 - 1j, -0.5 + 3j, -0.5 - 3j, -2]
        root_dots = Group(GlowDot(plane.n2p(root)) for root in roots)

        root_labels = VGroup(
            Tex(Rf"s_{{{n + 1}}}", font_size=36).next_to(dot.get_center(), UR, SMALL_BUFF)
            for n, dot in enumerate(root_dots)
        )
        root_labels.set_color(YELLOW)

        self.play(
            FadeIn(plane, lag_ratio=0.05),
            LaggedStart(*(FadeInFromPoint(dot, poly_rect.get_center()) for dot in root_dots), lag_ratio=0.3)
        )
        self.play(LaggedStartMap(FadeIn, root_labels, shift=0.25 * UP, lag_ratio=0.1))
        self.wait()

        # Show the solutions
        frame = self.frame
        axes = VGroup(
            Axes((0, 10), (-y_max, y_max), width=5, height=1.25)
            for root in roots
            for y_max in [3 if root.real > 0 else 1]
        )
        axes.arrange(DOWN, buff=0.75)
        axes.next_to(plane, RIGHT, buff=6)

        c_trackers = Group(ComplexValueTracker(1) for root in roots)
        graphs = VGroup(
            self.get_graph(axes, root, c_tracker.get_value)
            for axes, root, c_tracker in zip(axes, roots, c_trackers)
        )

        axes_labels = VGroup(
            Tex(Rf"e^{{s_{{{n + 1}}} t}}", font_size=60)
            for n in range(len(axes))
        )
        for label, ax in zip(axes_labels, axes):
            label.next_to(ax, LEFT, aligned_edge=UP)
            label[1:3].set_color(YELLOW)

        self.play(
            FadeIn(axes, lag_ratio=0.2),
            frame.animate.reorient(0, 0, 0, (4.67, -0.94, 0.0), 10.96),
            LaggedStart(
                (FadeTransform(m1.copy(), m2) for m1, m2 in zip(root_labels, axes_labels)),
                lag_ratio=0.05,
                group_type=Group
            ),
            LaggedStart(
                (ShowCreation(graph, suspend_mobject_updating=True) for graph in graphs),
                lag_ratio=0.05,
            ),
            run_time=3
        )
        self.wait()

        # Add on constants
        constant_labels = VGroup(
            Tex(Rf"c_{{{n + 1}}}", font_size=60).next_to(label[0], LEFT, SMALL_BUFF, aligned_edge=UP)
            for n, label in enumerate(axes_labels)
        )
        constant_labels.set_color(BLUE_B)
        self.play(
            LaggedStartMap(FadeIn, constant_labels, lag_ratio=0.1)
        )
        target_values = [0.5, 1j, 1.5, -1j, -1]
        self.play(LaggedStart(*(
            c_tracker.animate.set_value(value)
            for c_tracker, value in zip(c_trackers, target_values)
        )), lag_ratio=0.5, run_time=5)

        # Label as a solution
        solution_rect = SurroundingRectangle(VGroup(axes_labels, axes, constant_labels), buff=MED_SMALL_BUFF)
        solution_rect.set_stroke(WHITE, 1)
        solution_words = Text("All Solutions", font_size=60)
        solution_words.next_to(solution_rect, UP)
        solution_word = solution_words["Solutions"][0]
        solution_word.save_state(0)
        solution_word.match_x(solution_rect)

        const_rects = VGroup(SurroundingRectangle(c_label) for c_label in constant_labels)
        const_rects.set_stroke(BLUE, 3)

        plusses = Tex("+").replicate(4)
        for l1, l2, plus in zip(axes_labels, axes_labels[1:], plusses):
            plus.move_to(VGroup(l1, l2)).shift(SMALL_BUFF * LEFT)

        self.play(
            ShowCreation(solution_rect),
            Write(solution_word),
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeIn, plusses),
            Write(solution_words["All"]),
            Restore(solution_word),
        )
        self.wait()
        self.play(LaggedStartMap(ShowCreationThenFadeOut, const_rects, lag_ratio=0.25))
        self.play(LaggedStartMap(FadeOut, const_rects, lag_ratio=0.25))

    def get_graph(self, axes, s, get_const):
        def func(t):
            return (get_const() * np.exp(s * t)).real

        graph = axes.get_graph(func, bind=True, stroke_color=TEAL, stroke_width=2)
        return graph
