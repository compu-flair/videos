from manim_imports_ext import *


S_COLOR = YELLOW
T_COLOR = BLUE


class DefiningPropertyOfExp(InteractiveScene):
    def construct(self):
        # Key property
        tex_kw = dict(t2c={"{t}": GREY_B, "x": BLUE})
        equation = Tex(R"\frac{d}{d{t}} e^{t} = e^{t}", font_size=90, **tex_kw)

        exp_parts = equation["e^{t}"]
        ddt = equation[R"\frac{d}{d{t}}"]

        self.play(Write(exp_parts[0]))
        self.wait()
        self.play(FadeIn(ddt, scale=2))
        self.play(
            Write(equation["="]),
            TransformFromCopy(*exp_parts, path_arc=PI / 2),
        )
        self.wait()

        # Differential Equation
        ode = Tex(R"x'(t) = x(t)", font_size=72, **tex_kw)
        ode.move_to(equation).to_edge(UP)
        ode_label = Text("Differential\nEquation", font_size=36)
        ode_label.next_to(ode, LEFT, LARGE_BUFF, aligned_edge=DOWN)

        self.play(
            FadeTransform(equation.copy(), ode),
            FadeIn(ode_label)
        )
        self.wait()

        # Initial condition
        frame = self.frame
        abs_ic = Tex(R"x(0) = 1", font_size=72, **tex_kw)
        exp_ic = Tex(R"e^{0} = 1", font_size=90, t2c={"0": GREY_B})
        abs_ic.next_to(ode, RIGHT, buff=2.0)
        exp_ic.match_x(abs_ic).match_y(equation).shift(0.1 * UP)
        ic_label = Text("Initial\nCondition", font_size=36)
        ic_label.next_to(abs_ic, RIGHT, buff=0.75)

        self.play(
            FadeIn(abs_ic, RIGHT),
            FadeIn(exp_ic, RIGHT),
            frame.animate.set_x(2),
            Write(ic_label)
        )
        self.wait()

        # Scroll down
        self.play(frame.animate.set_y(-2.5), run_time=2)
        self.wait()


class ExampleExponentials(InteractiveScene):
    def construct(self):
        # Show the family
        pass

        # Highlight -1 + i term

        # Show e^t as its own derivative


class ImaginaryInputsToTheTaylorSeries(InteractiveScene):
    def construct(self):
        # Add complex plane
        plane = ComplexPlane(
            (-6, 6),
            (-4, 4),
            background_line_style=dict(stroke_color=BLUE, stroke_width=1),
            faded_line_style=dict(stroke_color=BLUE, stroke_width=0.5, stroke_opacity=0.5),
        )
        plane.set_height(5)
        plane.to_edge(DOWN, buff=0)
        plane.add_coordinate_labels(font_size=16)

        self.add(plane)

        # Add πi dot
        dot = GlowDot(color=YELLOW)
        dot.move_to(plane.n2p(PI * 1j))
        pi_i_label = Tex(R"\pi i", font_size=30).set_color(YELLOW)
        pi_i_label.next_to(dot, RIGHT, buff=-0.1).align_to(plane.n2p(3j), DOWN)

        self.add(dot, pi_i_label)

        # Show false equation
        false_eq = Tex(R"e^x = e \cdot e \cdots e \cdot e", t2c={"x": BLUE}, font_size=60)
        false_eq.to_edge(UP).shift(2 * LEFT)
        brace = Brace(false_eq[3:], DOWN)
        brace_tex = brace.get_tex(R"x \text{ times}")
        brace_tex[0].set_color(BLUE)

        nonsense = TexText(R"Nonsense if $x$ \\ is complex")
        nonsense.next_to(VGroup(false_eq, brace_tex), RIGHT, LARGE_BUFF)
        nonsense.set_color(RED)

        self.add(false_eq)
        self.play(GrowFromCenter(brace), FadeIn(brace_tex, lag_ratio=0.1))
        self.play(FadeIn(nonsense, lag_ratio=0.1))
        self.wait()

        # Make it the real equation
        gen_poly = self.get_series("x")
        gen_poly.to_edge(LEFT).to_edge(UP, MED_SMALL_BUFF)

        epii = self.get_series(R"\pi i", use_parens=True, in_tex_color=YELLOW)
        epii.next_to(gen_poly, DOWN, aligned_edge=LEFT)

        self.remove(false_eq)
        self.play(
            TransformFromCopy(false_eq[:2], gen_poly[0]),
            FadeOut(false_eq[2:], 0.5 * DOWN, lag_ratio=0.05),
            FadeOut(nonsense),
            FadeOut(brace, 0.5 * DOWN),
            FadeOut(brace_tex, 0.25 * DOWN),
            Write(gen_poly[1:])
        )
        self.wait()

        # Plug in πi
        vectors = self.get_spiral_vectors(plane, PI)
        buff = 0.5 * SMALL_BUFF
        labels = VGroup(
            Tex(R"\pi i", font_size=30).next_to(vectors[1], RIGHT, buff),
            Tex(R"(\pi^2 / 2) \cdot i^2", font_size=30).next_to(vectors[2], UP, buff),
            Tex(R"(\pi^3 / 6) \cdot i^3", font_size=30).next_to(vectors[3], LEFT, buff),
            Tex(R"(\pi^4 / 24) \cdot i^4", font_size=30).next_to(vectors[4], DOWN, buff),
        )
        labels.set_color(YELLOW)
        labels.set_backstroke(BLACK, 5)

        for n in range(0, len(gen_poly), 2):
            anims = [
                LaggedStart(
                    TransformMatchingTex(gen_poly[n].copy(), epii[n], run_time=1),
                    TransformFromCopy(gen_poly[n + 1], epii[n + 1]),
                    gen_poly[n + 2:].animate.align_to(epii[n + 2:], LEFT),
                    lag_ratio=0.05
                ),
            ]
            k = (n - 1) // 2
            if k >= 0:
                anims.append(GrowArrow(vectors[k]))
            if k == 1:
                anims.append(FadeTransform(pi_i_label, labels[0]))
            elif 2 <= k <= len(labels):
                anims.append(FadeIn(labels[k - 1]))
            if k >= 1:
                anims.append(dot.animate.set_width(0.5).move_to(vectors[k].get_end()))
            self.play(*anims)
        for vector in vectors[7:]:
            self.play(GrowArrow(vector), dot.animate.move_to(vector.get_end()))
        self.wait()

        # Step through terms individually
        labels.add_to_back(VectorizedPoint().move_to(vectors[0]))
        for n in range(5):
            rect1 = SurroundingRectangle(epii[2 * n + 2])
            rect2 = SurroundingRectangle(VGroup(vectors[n], labels[n]))
            self.play(
                FadeIn(rect1),
                self.fade_all_but(epii, 2 * n + 2),
                self.fade_all_but(vectors, n),
                self.fade_all_but(labels, n),
                dot.animate.set_opacity(0.1),
            )
            self.play(Transform(rect1, rect2))
            self.play(FadeOut(rect1))
        self.play(*(
            mob.animate.set_fill(opacity=1)
            for mob in [epii, vectors, labels]
        ))
        self.wait()

        # Swap out i for t
        e_to_it = self.get_series("it", use_parens=True, in_tex_color=GREEN)
        for sm1, sm2 in zip(e_to_it, epii):
            sm1.move_to(sm2)

        t_tracker = ValueTracker(PI)
        get_t = t_tracker.get_value

        t_label = Tex(R"t = 3.14", t2c={"t": GREEN})
        t_label.next_to(e_to_it, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)
        t_rhs = t_label.make_number_changeable("3.14")
        t_rhs.add_updater(lambda m: m.set_value(get_t()))

        vectors.add_updater(lambda m: m.become(self.get_spiral_vectors(plane, get_t(), 20)))
        dot.f_always.move_to(vectors[-1].get_end)

        max_theta = TAU
        semi_circle = Arc(0, max_theta, radius=plane.x_axis.get_unit_size(), arc_center=plane.n2p(0))
        semi_circle.set_stroke(TEAL, 3)

        self.play(
            ReplacementTransform(epii, e_to_it, lag_ratio=0.01, run_time=2),
            FadeOut(labels),
            FadeIn(t_label),
        )
        self.add(vectors)
        self.play(t_tracker.animate.set_value(0), run_time=5)
        self.play(
            t_tracker.animate.set_value(max_theta),
            ShowCreation(semi_circle),
            run_time=12
        )
        self.play(t_tracker.animate.set_value(PI), run_time=6)
        self.wait()

    def get_series(self, in_tex="x", use_parens=False, in_tex_color=BLUE, buff=0.2):
        paren_tex = f"({in_tex})" if use_parens else in_tex
        kw = dict(t2c={in_tex: in_tex_color})
        terms = VGroup(
            Tex(fR"e^{{{in_tex}}}", **kw),
            Tex("="),
            Tex(fR"1"),
            Tex(R"+"),
            Tex(fR"{in_tex}", **kw),
            Tex(R"+"),
            Tex(fR"\frac{{1}}{{2}} {paren_tex}^2", **kw),
            Tex(R"+"),
            Tex(fR"\frac{{1}}{{6}} {paren_tex}^3", **kw),
            Tex(R"+"),
            Tex(fR"\frac{{1}}{{24}} {paren_tex}^4", **kw),
            Tex(R"+"),
            Tex(R"\cdots", **kw),
            Tex(R"+"),
            Tex(fR"\frac{{1}}{{n!}} {paren_tex}^n", **kw),
            Tex(R"+"),
        )
        terms.arrange(RIGHT, buff=buff)
        terms[0].scale(1.25, about_edge=DR)
        return terms

    def get_spiral_vectors(
        self,
        plane,
        t,
        n_terms=10,
        # colors=[GREEN, YELLOW, GREEN_E, YELLOW_E]
        colors=[GREEN_E, GREEN_C, GREEN_B, GREEN_A],
    ):
        values = [(t * 1j)**n / math.factorial(n) for n in range(n_terms)]
        vectors = VGroup(
            Arrow(plane.n2p(0), plane.n2p(value), buff=0, fill_color=color)
            for value, color in zip(values, it.cycle(colors))
        )
        for v1, v2 in zip(vectors, vectors[1:]):
            v2.shift(v1.get_end() - v2.get_start())
        return vectors

    def fade_all_but(self, group, index):
        group.target = group.generate_target()
        group.target.set_fill(opacity=0.4)
        group.target[index].set_fill(opacity=1)
        return MoveToTarget(group)


class SPlane(InteractiveScene):
    tex_to_color_map = {"s": YELLOW, "t": BLUE, R"\omega": PINK}

    def construct(self):
        # Trackers
        s_tracker = self.s_tracker = ComplexValueTracker(-1)
        t_tracker = self.t_tracker = ValueTracker(0)
        get_s = s_tracker.get_value
        get_t = t_tracker.get_value

        # Add s plane
        s_plane = self.get_s_plane()
        s_dot, s_label = self.get_s_dot_and_label(s_plane, get_s)
        self.add(s_plane, s_dot, s_label)

        # Add exp plane
        exp_plane = self.get_exp_plane()
        exp_plane_label = self.get_exp_plane_label(exp_plane)
        output_dot, output_label = self.get_output_dot_and_label(exp_plane, get_s, get_t)
        output_path = self.get_output_path(exp_plane, get_t, get_s)

        self.add(exp_plane, exp_plane_label, output_path, output_dot, output_label)

        # Add e^{st} graph
        axes = self.get_graph_axes()
        graph = self.get_dynamic_exp_graph(axes, get_s)
        v_line = self.get_graph_v_line(axes, get_t, get_s)

        self.add(axes, graph, v_line)

        # Move s around, end at i
        s_tracker.set_value(-1)
        self.play(s_tracker.animate.set_value(0.2), run_time=4)
        self.play(s_tracker.animate.set_value(0), run_time=2)
        self.play(s_tracker.animate.set_value(1j), run_time=3)
        self.wait()

        # Let time tick forward
        frame = self.frame
        self.play_time_forward(
            3 * TAU,
            added_anims=[frame.animate.set_x(3).set_height(12).set_anim_args(time_span=(6, 15))],
        )
        self.wait()
        self.play(
            t_tracker.animate.set_value(0),
            frame.animate.set_x(1.5).set_height(10),
            run_time=3
        )

        # Set s to 2i, then add vectors
        t2c = {"2i": YELLOW, **self.tex_to_color_map}
        exp_2it_label = Tex(R"e^{2i t}", t2c=t2c)
        exp_2it_label.move_to(exp_plane_label, RIGHT)

        self.play(s_tracker.animate.set_value(2j), run_time=2)
        self.play(
            FadeOut(exp_plane_label, 0.5 * UP),
            FadeIn(exp_2it_label, 0.5 * UP),
        )
        self.play_time_forward(TAU)

        # Show the derivative
        exp_plane.target = exp_plane.generate_target()
        exp_plane.target.align_to(axes.c2p(0, 0), LEFT)

        deriv_expression = Tex(R"\frac{d}{dt} e^{2i t} = 2i \cdot e^{2i t}", t2c=t2c)
        deriv_expression.next_to(exp_plane.target, RIGHT, aligned_edge=UP)

        self.play(
            LaggedStart(
                ReplacementTransform(exp_2it_label, deriv_expression["e^{2i t}"][0], path_arc=-90 * DEG),
                MoveToTarget(exp_plane),
                Write(deriv_expression[R"\frac{d}{dt}"]),
                lag_ratio=0.2
            ),
            frame.animate.reorient(0, 0, 0, (1, 0, 0.0), 9.25),
        )
        self.play(LaggedStart(
            Write(deriv_expression["="]),
            TransformFromCopy(*deriv_expression["e^{2i t}"], path_arc=-90 * DEGREES),
            FadeTransform(deriv_expression["e^{2i t}"][0][1:3].copy(), deriv_expression[R"2i"][1], path_arc=-90 * DEG),
            Write(deriv_expression[R"\cdot"]),
            lag_ratio=0.25,
            run_time=1.5
        ))
        self.add(deriv_expression)
        self.wait()

        # Step through derivative parts
        v_part, p_part, i_part, two_part = parts = VGroup(
            deriv_expression[R"\frac{d}{dt} e^{2i t}"][0],
            deriv_expression[R"e^{2i t}"][1],
            deriv_expression[R"i"][1],
            deriv_expression[R"2"][1],
        )
        colors = [GREEN, BLUE, YELLOW, YELLOW]
        labels = VGroup(Text("Velocity"), Text("Position"), Tex(R"90^{\circ}"), Text("Stretch"))
        for part, color, label in zip(parts, colors, labels):
            part.rect = SurroundingRectangle(part, buff=0.05)
            part.rect.set_stroke(color, 2)
            label.set_color(color)
            label.next_to(part.rect, DOWN)
            part.label = label

        p_vect, v_vect = vectors = self.get_pv_vectors(exp_plane, get_t, get_s)
        for vector in vectors:
            vector.suspend_updating()
        p_vect_copy = p_vect.copy().clear_updaters()

        self.play(
            ShowCreation(v_part.rect),
            FadeIn(v_part.label),
            GrowArrow(v_vect),
        )
        self.wait()
        self.play(
            ReplacementTransform(v_part.rect, p_part.rect),
            FadeTransform(v_part.label, p_part.label),
            GrowArrow(p_vect),
        )
        self.wait()
        self.play(
            ReplacementTransform(p_part.rect, i_part.rect),
            FadeTransform(p_part.label, i_part.label),
            p_vect_copy.animate.rotate(90 * DEG, about_point=exp_plane.n2p(0)).shift(p_vect.get_vector())
        )
        self.wait()
        self.play(
            ReplacementTransform(i_part.rect, two_part.rect),
            FadeTransform(i_part.label, two_part.label),
            Transform(p_vect_copy, v_vect, remover=True)
        )
        self.wait()
        self.play(FadeOut(two_part.rect), FadeOut(two_part.label))

        vectors.resume_updating()
        self.play_time_forward(TAU)

        # Label this angular frequency with omega
        imag_exp = Tex(R"e^{i \omega t}", t2c=self.tex_to_color_map, font_size=60)
        imag_exp.move_to(deriv_expression, LEFT)

        self.play(
            FadeOut(deriv_expression, 0.5 * UP),
            FadeIn(imag_exp, 0.5 * UP),
        )
        t_tracker.set_value(0)
        output_path.suspend_updating()
        self.play(s_tracker.animate.set_value(1.5j), run_time=3)
        output_path.resume_updating()
        self.play_time_forward(TAU * 4 / 3)

        # Move to other complex values, end at -0.5 + i
        t_max_tracker = ValueTracker(20 * TAU)
        new_output_path = self.get_output_path(exp_plane, t_max_tracker.get_value, get_s)
        output_path.match_updaters(new_output_path)
        t_tracker.set_value(0)

        self.play(
            FadeOut(imag_exp, time_span=(0, 1)),
            s_tracker.animate.set_value(-0.2 + 1.5j),
            run_time=2
        )
        self.play(s_tracker.animate.set_value(-0.2 + 1j), run_time=2)
        self.play(s_tracker.animate.set_value(-0.5 + 1j), run_time=2)
        self.play_time_forward(TAU)

        # Split up the exponential to e^{-0.5t} * e^{it}
        t2c = {"-0.5": YELLOW, "i": YELLOW, "t": BLUE}
        lines = VGroup(
            Tex(R"e^{(-0.5 + i)t}", t2c=t2c),
            Tex(R"\left(e^{-0.5t} \right) \left(e^{it} \right)", t2c=t2c)
        )
        lines.arrange(DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)
        lines.next_to(exp_plane, RIGHT, LARGE_BUFF, aligned_edge=UP)

        dec_brace = Brace(lines[1][R"\left(e^{-0.5t} \right)"], DOWN, SMALL_BUFF)
        rot_brace = Brace(lines[1][R"\left(e^{it} \right)"], DOWN, SMALL_BUFF)
        dec_label = dec_brace.get_text("Decay")
        rot_label = rot_brace.get_text("Rotation")

        self.play(
            FadeIn(lines[0], time_span=(0.5, 1)),
            FadeTransform(s_label[-1].copy(), lines[0]["-0.5 + i"])
        )
        self.wait()
        self.play(
            TransformMatchingTex(lines[0].copy(), lines[1], run_time=1, lag_ratio=0.01)
        )
        self.wait()
        self.play(
            GrowFromCenter(dec_brace),
            FadeIn(dec_label)
        )
        self.wait()
        self.play(
            ReplacementTransform(dec_brace, rot_brace),
            FadeTransform(dec_label, rot_label),
        )
        self.wait()
        self.play(
            FadeOut(rot_brace),
            FadeOut(rot_label),
            t_tracker.animate.set_value(0).set_anim_args(run_time=3)
        )

        # Show multiplication by s
        s_vect = Arrow(s_plane.n2p(0), s_plane.n2p(get_s()), buff=0, fill_color=YELLOW)
        one_vect = Arrow(s_plane.n2p(0), s_plane.n2p(1), buff=0, fill_color=BLUE)
        arc = Line(
            s_plane.n2p(0.3),
            s_plane.n2p(0.3 * get_s()),
            path_arc=s_vect.get_angle(),
            buff=0.1
        )
        arc.add_tip(length=0.25, width=0.25)
        times_s_label = Tex(R"\times s")
        times_s_label.next_to(arc.pfp(0.5), UR, SMALL_BUFF)

        self.play(LaggedStart(
            FadeIn(one_vect),
            FadeIn(arc),
            FadeIn(times_s_label),
            FadeIn(s_vect),
        ))
        self.wait()
        self.play(
            TransformFromCopy(one_vect, s_vect, path_arc=s_vect.get_angle()),
            run_time=2
        )
        self.wait()
        self.play(
            TransformFromCopy(one_vect, p_vect),
            TransformFromCopy(s_vect, v_vect),
            run_time=2
        )

        # Show spiraling inward
        self.play_time_forward(2 * TAU)

        self.play(FadeOut(VGroup(arc, times_s_label, lines)))
        t_tracker.set_value(0)

        s_vect.add_updater(lambda m: m.put_start_and_end_on(s_plane.n2p(0), s_plane.n2p(get_s())))
        self.add(s_vect)

        # Tour various values on the s plane
        values = [
            -0.1 - 2j,
            -0.1 + 2j,
            -0.1 + 0.5j,
            +0.1 + 0.5j,
            -0.5 + 0.5j,
            -0.1 + 0.5j,
        ]
        for value in values:
            self.play(s_tracker.animate.set_value(value), run_time=5)

        self.play_time_forward(4 * TAU)

    def get_s_plane(self):
        s_plane = ComplexPlane((-2, 2), (-2, 2))
        s_plane.set_width(7)
        s_plane.to_edge(LEFT, buff=SMALL_BUFF)
        s_plane.add_coordinate_labels(font_size=16)
        return s_plane

    def get_s_dot_and_label(self, s_plane, get_s):
        s_dot = Group(
            Dot(radius=0.05, fill_color=YELLOW),
            GlowDot(color=YELLOW),
        )
        s_dot.add_updater(lambda m: m.move_to(s_plane.n2p(get_s())))

        s_label = Tex(R"s = +0.5", font_size=36)
        s_rhs = s_label.make_number_changeable("+0.5", include_sign=True)
        s_rhs.f_always.set_value(get_s)
        s_label.set_color(S_COLOR)
        s_label.set_backstroke(BLACK, 5)
        s_label.always.next_to(s_dot[0], UR, SMALL_BUFF)

        return Group(s_dot, s_label)

    def get_exp_plane(self):
        tex_kw = dict(t2c={"s": YELLOW, "t": BLUE})

        exp_plane = ComplexPlane((-2, 2), (-2, 2))
        exp_plane.background_lines.set_stroke(width=1)
        exp_plane.faded_lines.set_stroke(opacity=0.25)
        exp_plane.set_width(4)
        exp_plane.to_corner(DR).shift(0.5 * LEFT)

        return exp_plane

    def get_exp_plane_label(self, exp_plane):
        label = Tex(R"e^{st}", font_size=60, t2c=self.tex_to_color_map)
        label.set_backstroke(BLACK, 5)
        label.next_to(exp_plane.get_corner(UL), DL, 0.2)
        return label

    def get_output_dot_and_label(self, exp_plane, get_s, get_t):
        output_dot = GlowDot(color=GREEN)
        output_dot.add_updater(lambda m: m.move_to(exp_plane.n2p(np.exp(get_s() * get_t()))))

        output_label = Tex(R"e^{s \cdot 0.00}", font_size=36, t2c=self.tex_to_color_map)
        t_label = output_label.make_number_changeable("0.00")
        t_label.set_color(BLUE)
        t_label.match_height(output_label["s"], about_edge=LEFT)
        t_label.f_always.set_value(get_t)
        output_label.always.next_to(output_dot, UP, SMALL_BUFF, LEFT).shift(0.2 * DR)
        output_label.set_backstroke(BLACK, 3)

        return Group(output_dot, output_label)

    def get_graph_axes(
        self,
        x_range=(0, 24),
        y_range=(-2, 2),
        width=15,
        height=2,
    ):
        axes = Axes(x_range, y_range, width=width, height=height)
        x_axis_label = Tex(R"t", font_size=36, t2c=self.tex_to_color_map)
        y_axis_label = Tex(R"\text{Re}\left[e^{st}\right]", font_size=36, t2c=self.tex_to_color_map)
        x_axis_label.next_to(axes.x_axis.get_right(), UP, buff=0.15)
        y_axis_label.next_to(axes.y_axis.get_top(), UP, SMALL_BUFF)
        axes.add(x_axis_label)
        axes.add(y_axis_label)
        axes.next_to(ORIGIN, RIGHT, MED_LARGE_BUFF)
        axes.to_edge(UP, buff=0.5)
        x_axis_label.shift_onto_screen(buff=MED_LARGE_BUFF)
        return axes

    def get_dynamic_exp_graph(self, axes, get_s, delta_t=0.1, stroke_color=TEAL, stroke_width=3):
        graph = Line().set_stroke(stroke_color, stroke_width)
        t_samples = np.arange(*axes.x_range[:2], 0.1)

        def update_graph(graph):
            s = get_s()
            values = np.exp(s * t_samples)
            xs = values.astype(np.complex128).real
            graph.set_points_smoothly(axes.c2p(t_samples, xs))

        graph.add_updater(update_graph)
        return graph

    def get_graph_v_line(self, axes, get_t, get_s):
        v_line = Line(DOWN, UP)
        v_line.set_stroke(WHITE, 2)
        v_line.f_always.put_start_and_end_on(
            lambda: axes.c2p(get_t(), 0),
            lambda: axes.c2p(get_t(), np.exp(get_s() * get_t()).real),
        )
        return v_line

    def get_output_path(self, exp_plane, get_t, get_s, delta_t=1 / 30, color=TEAL, stroke_width=2):
        path = VMobject()
        path.set_points([ORIGIN])
        path.set_stroke(color, stroke_width)

        def get_path_points():
            t_range = np.arange(0, get_t(), delta_t)
            if len(t_range) == 0:
                t_range = np.array([0])
            values = np.exp(t_range * get_s())
            return np.array([exp_plane.n2p(z) for z in values])

        path.f_always.set_points_smoothly(get_path_points)
        return path

    def play_time_forward(self, time, added_anims=[]):
        self.t_tracker.set_value(0)
        self.play(
            self.t_tracker.animate.set_value(time).set_anim_args(rate_func=linear),
            *added_anims,
            run_time=time,
        )

    def get_pv_vectors(self, exp_plane, get_t, get_s, thickness=3, colors=[BLUE, YELLOW]):
        p_vect = Vector(RIGHT, fill_color=colors[0], thickness=thickness)
        v_vect = Vector(RIGHT, fill_color=colors[1], thickness=thickness)
        p_vect.add_updater(lambda m: m.put_start_and_end_on(
            exp_plane.n2p(0),
            exp_plane.n2p(np.exp(get_t() * get_s()))
        ))
        v_vect.add_updater(lambda m: m.put_start_and_end_on(
            exp_plane.n2p(0),
            exp_plane.n2p(get_s() * np.exp(get_t() * get_s()))
        ).shift(p_vect.get_vector()))

        return VGroup(p_vect, v_vect)

    ###

    def old_material(self):
        # Collapse the graph
        output_dot = GlowDot(color=GREEN)
        output_dot.move_to(axes.x_axis.n2p(1))
        output_label = Tex(R"e^{s \cdot 0.00}", **tex_kw, font_size=36)
        t_tracker.set_value(0)
        t_label = output_label.make_number_changeable("0.00")
        t_label.set_color(BLUE)
        t_label.match_height(output_label["s"], about_edge=LEFT)
        t_label.f_always.set_value(get_t)
        output_label.add_updater(lambda m: m.next_to(output_dot, UP, SMALL_BUFF, LEFT).shift(0.2 * DR))

        graph.clear_updaters()
        self.remove(axes)
        self.play(LaggedStart(
            FadeOut(VGroup(axes.x_axis, x_axis_label, graph)),
            AnimationGroup(
                Rotate(axes.y_axis, -90 * DEG),
                TransformMatchingTex(y_axis_label, output_label, run_time=1.5),
            ),
            FadeIn(output_dot),
            lag_ratio=0.5
        ))
