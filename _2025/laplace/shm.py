from manim_imports_ext import *


class SrpingMassSystem(VGroup):
    def __init__(
        self,
        x0=0,
        v0=0,
        k=3,
        mu=0.1,
        equilibrium_length=7,
        equilibrium_position=UP,
        spring_stroke_color=GREY_B,
        spring_stroke_width=2,
        spring_radius=0.25,
        n_spring_curls=8,
        mass_width=1.0,
        mass_color=BLUE_E,
        mass_label="m",
    ):
        super().__init__()
        self.equilibrium_position = equilibrium_position
        self.fixed_spring_point = equilibrium_position + (equilibrium_length - 0.5 * mass_width) * LEFT
        self.mass = self.get_mass(mass_width, mass_color, mass_label)
        self.spring = self.get_spring(spring_stroke_color, spring_stroke_width, n_spring_curls, spring_radius)
        self.add(self.spring, self.mass)

        self.set_x(x0)
        self.velocity = v0

    def get_spring(self, stroke_color, stroke_width, n_curls, radius):
        spring = ParametricCurve(
            lambda t: [t, -radius * math.sin(TAU * t), radius * math.cos(TAU * t)],
            t_range=(0, n_curls, 1e-2),
            stroke_color=stroke_color,
            stroke_width=stroke_width,
        )
        return spring

    def get_mass(self, mass_width, mass_color, mass_label):
        mass = Square(mass_width)
        mass.set_fill(mass_color, 1)
        mass.set_stroke(WHITE, 1)
        mass.set_shading(0.1, 0.1, 0.1)
        label = Tex(mass_label)
        label.set_max_width(0.5 * mass.get_width())
        label.move_to(mass)
        mass.add(label)
        return mass

    def set_x(self, x):
        self.mass.move_to(self.equilibrium_position + x * RIGHT)
        spring_width = SMALL_BUFF + get_norm(self.mass.get_left() - self.fixed_spring_point)
        self.spring.set_width(spring_width, stretch=True)
        self.spring.move_to(self.fixed_spring_point, LEFT)

    def get_x(self):
        return (self.mass.get_center() - self.equilibrium_position)[0]

    def time_step(self, delta_t):
        pass


class BasicSpringScene(InteractiveScene):
    def construct(self):
        # Add spring, give some initial oscillation
        spring = SrpingMassSystem()
        self.add(spring)

        spring.add_updater(lambda m: m.set_x(2 * math.cos(TAU * self.time) * math.exp(-0.5 * self.time)))
        self.wait(10)

        # Label on a number line

        # Show the force law

        # Write the differential equation

        # Show the solution graph

        # Add damping term


class SolveDampedSpringEquation(InteractiveScene):
    def construct(self):
        # Show x and its derivatives
        pos, vel, acc = funcs = VGroup(
            Tex(R"x(t)"),
            Tex(R"x'(t)"),
            Tex(R"x''(t)"),
        )
        funcs.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)

        labels = VGroup(
            Text("Position").set_color(BLUE),
            Text("Velocity").set_color(RED),
            Text("Acceleration").set_color(YELLOW),
        )
        colors = [interpolate_color_by_hsl(TEAL, RED, a) for a in np.linspace(0, 1, 3)]
        for line, label, color in zip(funcs, labels, colors):
            label.set_color(color)
            label.next_to(line, RIGHT, MED_LARGE_BUFF)
            label.align_to(labels[0], LEFT)

        VGroup(funcs, labels).to_corner(UR)

        arrows = VGroup()
        for l1, l2 in zip(funcs, funcs[1:]):
            arrow = Line(l1.get_left(), l2.get_left(), path_arc=150 * DEG, buff=0.2)
            arrow.add_tip(width=0.2, length=0.2)
            arrow.set_color(GREY_B)
            ddt = Tex(R"\frac{d}{dt}", font_size=30)
            ddt.set_color(GREY_B)
            ddt.next_to(arrow, LEFT, SMALL_BUFF)
            arrow.add(ddt)
            arrows.add(arrow)

        self.play(Write(funcs[0]), Write(labels[0]))
        self.wait()
        for func1, func2, label1, label2, arrow in zip(funcs, funcs[1:], labels, labels[1:], arrows):
            self.play(LaggedStart(
                GrowFromPoint(arrow, arrow.get_corner(UR), path_arc=30 * DEG),
                TransformFromCopy(func1, func2, path_arc=30 * DEG),
                FadeTransform(label1.copy(), label2),
                lag_ratio=0.1
            ))
            self.wait()

        deriv_group = VGroup(funcs, labels, arrows)

        # Show F=ma
        t2c = {
            "x(t)": colors[0],
            "x'(t)": colors[1],
            "x''(t)": colors[2],
        }
        equation1 = Tex(R"m x''(t) = -k x(t) - \mu x'(t)", t2c=t2c)
        equation1.to_corner(UL)

        ma = equation1["m x''(t)"][0]
        kx = equation1["-k x(t)"][0]
        mu_v = equation1[R"- \mu x'(t)"][0]
        rhs = VGroup(kx, mu_v)

        ma_brace, kx_brace, mu_v_brace = braces = VGroup(
            Brace(part, DOWN, buff=SMALL_BUFF)
            for part in [ma, kx, mu_v]
        )
        label_texs = [R"\textbf{F}", R"\text{Spring force}", R"\text{Damping}"]
        for brace, label_tex in zip(braces, label_texs):
            brace.add(brace.get_tex(label_tex))

        self.play(TransformFromCopy(acc, ma[1:], path_arc=-45 * DEG))
        self.play(LaggedStart(
            GrowFromCenter(ma_brace),
            Write(ma[0]),
            run_time=1,
            lag_ratio=0.1
        ))
        self.wait()
        self.play(LaggedStart(
            Write(equation1["= -k"][0]),
            FadeTransformPieces(ma_brace, kx_brace),
            TransformFromCopy(pos, equation1["x(t)"][0], path_arc=-45 * DEG),
        ))
        self.wait()
        self.play(LaggedStart(
            FadeTransformPieces(kx_brace, mu_v_brace),
            Write(equation1[R"- \mu"][0]),
            TransformFromCopy(vel, equation1["x'(t)"][0], path_arc=-45 * DEG),
        ))
        self.wait()
        self.play(FadeOut(mu_v_brace))

        # Rearrange
        equation2 = Tex(R"m x''(t) + \mu x'(t) + k x(t) = 0", t2c=t2c)
        equation2.move_to(equation1, UL)

        self.play(TransformMatchingTex(equation1, equation2, path_arc=45 * DEG))
        self.wait()

        # Hypothesis of e^st
        t2c = {"s": YELLOW, "x(t)": TEAL}
        hyp_word, hyp_tex = hypothesis = VGroup(
            Text("Hypothesis: "),
            Tex("x(t) = e^{st}", t2c=t2c),
        )
        hypothesis.arrange(RIGHT)
        hypothesis.to_corner(UR)
        sub_hyp_word = TexText(R"(For some $s$)", t2c={"$s$": YELLOW}, font_size=36, fill_color=GREY_B)
        sub_hyp_word.next_to(hyp_tex, DOWN)

        self.play(LaggedStart(
            FadeTransform(pos.copy(), hyp_tex[:4], path_arc=45 * DEG, remover=True),
            FadeOut(deriv_group),
            Write(hyp_word, run_time=1),
            Write(hyp_tex[4:], time_span=(0.5, 1.5)),
        ))
        self.add(hypothesis)
        self.wait()
        self.play(FadeIn(sub_hyp_word, 0.25 * DOWN))
        self.wait()

        # Plug it in
        t2c["s"] = YELLOW
        equation3 = Tex(R"m s^2 e^{st} + \mu s e^{st} + k e^{st} = 0", t2c=t2c)
        equation3.next_to(equation2, DOWN, LARGE_BUFF)
        pos_parts = VGroup(equation2["x(t)"][0], equation3["e^{st}"][-1])
        vel_parts = VGroup(equation2["x'(t)"][0], equation3["s e^{st}"][0])
        acc_parts = VGroup(equation2["x''(t)"][0], equation3["s^2 e^{st}"][0])
        matched_parts = VGroup(pos_parts, vel_parts, acc_parts)

        pos_rect, vel_rect, acc_rect = rects = VGroup(
            SurroundingRectangle(group[0], buff=0.05).set_stroke(group[0][0].get_color(), 1)
            for group in matched_parts
        )

        pos_arrow, vel_arrow, acc_arrow = arrows = VGroup(
            Arrow(*pair, buff=0.1)
            for pair in matched_parts
        )

        for rect, arrow, pair in zip(rects, arrows, matched_parts):
            self.play(ShowCreation(rect))
            self.play(
                GrowArrow(arrow),
                FadeTransform(pair[0].copy(), pair[1]),
                rect.animate.surround(pair[1]),
            )
            self.wait()
        self.play(
            LaggedStart(
                (TransformFromCopy(equation2[tex], equation3[tex])
                for tex in ["m", "+", "k", R"\mu", "=", "0"]),
                lag_ratio=0.05,
            ),
        )
        self.wait()
        self.play(FadeOut(arrows, lag_ratio=0.1), FadeOut(rects, lag_ratio=0.1))

        # Solve for s
        key_syms = ["s", "m", R"\mu", "k"]
        equation4, equation5, equation6 = new_equations = VGroup(
            Tex(R"e^{st} \left( ms^2 + \mu s + k \right) = 0", t2c=t2c),
            Tex(R"ms^2 + \mu s + k = 0", t2c=t2c),
            Tex(R"{s} = {{-\mu \pm \sqrt{\mu^2 - 4mk}} \over 2m}", isolate=key_syms)
        )
        rhs = equation6[2:]
        rhs.set_width(equation5.get_width() - equation6[:2].get_width(), about_edge=LEFT)
        equation6.refresh_bounding_box()
        equation6["{s}"].set_color(YELLOW)
        new_equations.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        new_equations.move_to(equation3, UL)
        equation4 = new_equations[0]

        exp_rect = SurroundingRectangle(equation4[R"e^{st}"])
        exp_rect.set_stroke(YELLOW, 2)
        ne_0 = VGroup(Tex(R"\ne").rotate(90 * DEG), Integer(0))
        ne_0.arrange(DOWN).next_to(exp_rect, DOWN)

        self.play(
            TransformMatchingTex(
                equation3,
                equation4,
                matched_keys=[R"e^{st}"],
                run_time=1.5,
                path_arc=30 * DEG
            )
        )
        self.wait(0.5)
        self.play(ShowCreation(exp_rect))
        self.wait()
        self.play(Write(ne_0))
        self.wait()
        self.play(FadeOut(ne_0))
        self.play(
            *(
                TransformFromCopy(equation4[key], equation5[key])
                for key in [R"ms^2 + \mu s + k", "= 0"]
            ),
            FadeOut(exp_rect),
        )
        self.wait()

        # Show mirror image
        self.play(
            TransformMatchingTex(
                equation5.copy(), equation2.copy(),
                key_map={
                    "s^2": "x''(t)",
                    R"\mu s": R"\mu x(t)",
                    R"k": R"k x(t)",
                },
                # match_animation=FadeTransform,
                # mismatch_animation=FadeTransform,
                remover=True,
                rate_func=there_and_back_with_pause,
                run_time=6
            ),
            equation4.animate.set_fill(opacity=0.25),
        )
        self.play(equation4.animate.set_fill(opacity=1))
        self.wait()

        # Show quadratic formula
        qf_arrow = Arrow(
            equation5.get_right(),
            equation6.get_right(),
            path_arc=-120 * DEG
        )
        qf_words = Text("Quadratic\nFormula", font_size=30, fill_color=GREY_B)
        qf_words.next_to(qf_arrow, RIGHT)

        naked_equation = equation6.copy()
        for sym in key_syms:
            naked_equation[sym].scale(0).set_fill(opacity=0).move_to(naked_equation.get_right())

        self.play(
            TransformFromCopy(equation5["s"], equation6["s"]),
            Write(equation6["="]),
            GrowFromPoint(qf_arrow, qf_arrow.get_corner(UL)),
            FadeIn(qf_words, shift=0.5 * DOWN),
            # Write(naked_equation),
        )
        self.play(
            LaggedStart(*(
                TransformFromCopy(equation5[sym], equation6[sym], time_span=(0.5, 1.5))
                for sym in key_syms[1:]
            ), lag_ratio=0.1),
            Write(naked_equation),
        )
        self.wait()
        self.remove(naked_equation)
        self.add(equation6)

        # Move hypothesis
        frame = self.frame
        equation6.target = equation6.generate_target()
        equation6.target.scale(1.25, about_edge=LEFT).shift(0.5 * DOWN)
        qf_rect = SurroundingRectangle(equation6.target[2:])
        qf_rect.set_stroke(YELLOW, 1.5)
        self.play(
            hypothesis.animate.next_to(equation2, UP, LARGE_BUFF, aligned_edge=LEFT),
            FadeOut(sub_hyp_word),
            frame.animate.scale(1.5, about_edge=LEFT),
            FadeOut(qf_arrow),
            FadeOut(qf_words),
            MoveToTarget(equation6),
            ShowCreation(qf_rect, time_span=(0.75, 1.5)),
            run_time=1.5
        )
        self.wait()

    def old_material(self):
        # Show implied exponentials
        final_equation = new_equations[-1]
        consolidated_lines = VGroup(
            hypothesis,
            equation2,
            equation4,
            final_equation,
        )
        consolidated_lines.target = consolidated_lines.generate_target()
        consolidated_lines.target.scale(0.7)
        consolidated_lines.target.arrange(DOWN, buff=MED_LARGE_BUFF)
        consolidated_lines.target.to_corner(UL)

        implies = Tex(R"\Longrightarrow", font_size=60)
        implies.next_to(consolidated_lines.target[0], RIGHT, buff=0.75)

        t2c = {"x(t)": TEAL, R"\omega": PINK}
        imag_exps = VGroup(
            Tex(R"x(t) = e^{+i \omega t}", t2c=t2c),
            Tex(R"x(t) = e^{-i \omega t}", t2c=t2c),
        )
        imag_exps.arrange(RIGHT, buff=2.0)
        imag_exps.next_to(implies, RIGHT, buff=0.75)

        self.remove(final_equation)
        self.play(LaggedStart(
            FadeOut(arrows),
            FadeOut(equation3, 0.5 * UP),
            FadeOut(sub_hyp_word),
            MoveToTarget(consolidated_lines),
            Write(implies),
        ))
        for imag_exp, sgn in zip(imag_exps, "+-"):
            self.play(
                TransformFromCopy(hyp_tex["x(t) ="][0], imag_exp["x(t) ="][0]),
                TransformFromCopy(hyp_tex["e"][0], imag_exp["e"][0]),
                TransformFromCopy(hyp_tex["t"][-1], imag_exp["t"][-1]),
                FadeTransform(final_equation[R"\pm i"][0].copy(), imag_exp[Rf"{sgn}i"][0]),
                FadeTransform(final_equation[R"\sqrt{k/m}"][0].copy(), imag_exp[R"\omega"][0]),
            )

        omega_brace = Brace(final_equation[R"\sqrt{k/m}"], DOWN, SMALL_BUFF)
        omega_label = omega_brace.get_tex(R"\omega").set_color(PINK)
        self.play(GrowFromCenter(omega_brace), Write(omega_label))
        self.wait()

        # Combine two solutions
        cos_equation = Tex(R"e^{+i \omega t} + e^{-i \omega t} = 2\cos(\omega t)", t2c={R"\omega": PINK})
        cos_equation.move_to(imag_exps)
        omega_brace2 = omega_brace.copy()
        omega_brace2.stretch(0.5, 0).match_width(cos_equation[R"\omega"][-1])
        omega_brace2.next_to(cos_equation[R"\omega"][-1], DOWN, SMALL_BUFF)
        omega_brace2_tex = omega_brace2.get_tex(R"\sqrt{k / m}", buff=SMALL_BUFF, font_size=24)

        self.remove(imag_exps)
        self.play(
            TransformFromCopy(imag_exps[0][R"e^{+i \omega t}"], cos_equation[R"e^{+i \omega t}"]),
            TransformFromCopy(imag_exps[1][R"e^{-i \omega t}"], cos_equation[R"e^{-i \omega t}"]),
            FadeOut(imag_exps[0][R"x(t) ="]),
            FadeOut(imag_exps[1][R"x(t) ="]),
            Write(cos_equation["+"][1]),
        )
        self.wait()
        self.play(Write(cos_equation[R"= 2\cos(\omega t)"]))
        self.wait()
        self.play(GrowFromCenter(omega_brace2), Write(omega_brace2_tex))

        # Clear the board
        self.play(LaggedStart(
            FadeOut(implies),
            FadeOut(cos_equation),
            FadeOut(omega_brace2),
            FadeOut(omega_brace2_tex),
            FadeOut(consolidated_lines[2:]),
            FadeOut(omega_brace),
            FadeOut(omega_label),
            lag_ratio=0.1
        ))

        # Add damping term
        t2c = {"x''(t)": colors[2], "x'(t)": colors[1], "x(t)": colors[0], "{s}": YELLOW}
        new_lines = VGroup(
            Tex(R"m x''(t) + \mu x'(t) + k x(t) = 0", t2c=t2c),
            Tex(R"m ({s}^2 e^{{s}t}) + \mu ({s} e^{{s}t}) + k (e^{{s}t}) = 0", t2c=t2c),
            Tex(R"e^{{s}t}\left(m {s}^2 + \mu {s} + k \right) = 0", t2c=t2c),
            Tex(R"m {s}^2 + \mu {s} + k = 0", t2c=t2c),
            Tex(R"{s} = {{-\mu \pm \sqrt{\mu^2 - 4mk}} \over 2m}", t2c=t2c),
        )
        new_lines.scale(0.7)
        new_lines.arrange(DOWN, aligned_edge=LEFT, buff=MED_LARGE_BUFF)
        new_lines.move_to(equation2, UL)

        self.play(
            TransformMatchingTex(
                equation2,
                new_lines[0],
                matched_keys=t2c.keys(),
                run_time=1
            )
        )
        self.wait()
        for line1, line2 in zip(new_lines, new_lines[1:]):
            if line1 is new_lines[0]:
                key_map = {
                    "x''(t)": R"({s}^2 e^{{s}t})",
                    "x'(t)": R"({s} e^{{s}t})",
                    "x(t)": R"(e^{{s}t})",
                }
            else:
                key_map = dict()
            self.play(TransformMatchingTex(line1.copy(), line2, key_map=key_map, run_time=1, lag_ratio=0.01))
            self.wait()


class DampedSpringSolutionsOnSPlane(InteractiveScene):
    def construct(self):
        # Background
        self.add_background_image()

        # Add the plane
        plane = ComplexPlane((-3, 2), (-2, 2))
        plane.set_height(5)
        plane.background_lines.set_stroke(BLUE, 1)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)
        plane.add_coordinate_labels(font_size=24)
        plane.move_to(DOWN)
        plane.to_edge(RIGHT, buff=1.0)
        self.add(plane)

        # Add the sliders
        colors = [interpolate_color_by_hsl(RED, TEAL, a) for a in np.linspace(0, 1, 3)]
        chars = ["m", R"\mu", "k"]
        m_slider, mu_slider, k_slider = sliders = VGroup(
            self.get_slider(char, color)
            for char, color in zip(chars, colors)
        )
        m_tracker, mu_tracker, k_tracker = trackers = Group(
            slider.value_tracker for slider in sliders
        )
        sliders.arrange(RIGHT, buff=MED_LARGE_BUFF)
        sliders.next_to(plane, UP, aligned_edge=LEFT)

        for tracker, value in zip(trackers, [1, 0, 0]):
            tracker.set_value(value)

        self.add(trackers)
        self.add(sliders)

        # Add the dots
        def get_roots():
            a, b, c = [t.get_value() for t in trackers]
            m = -b / 2
            p = c / a
            disc = m**2 - p
            radical = math.sqrt(disc) if disc >= 0 else 1j * math.sqrt(-disc)
            return (m + radical, m - radical)

        def update_dots(dots):
            roots = get_roots()
            for dot, root in zip(dots, roots):
                dot.move_to(plane.n2p(root))

        root_dots = GlowDot().replicate(2)
        root_dots.add_updater(update_dots)

        rect_edge_point = (-3.33, -1.42, 0.0)

        def update_lines(lines):
            for line, dot in zip(lines, root_dots):
                line.put_start_and_end_on(rect_edge_point, dot.get_center())

        lines = Line().replicate(2)
        lines.set_stroke(YELLOW, 2, 0.35)
        lines.add_updater(update_lines)

        self.add(root_dots)
        self.add(lines)

        # Play with values
        self.play(k_tracker.animate.set_value(2), run_time=3)
        self.wait()
        self.play(mu_tracker.animate.set_value(2), run_time=3)
        self.wait()
        self.play(k_tracker.animate.set_value(5), run_time=3)
        self.wait()
        self.play(k_tracker.animate.set_value(2), run_time=3)

        # Zoom out and show graph
        frame = self.frame

        axes = Axes((0, 10, 1), (-1, 1, 1), width=10, height=3.5)
        axes.next_to(plane, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)

        def func(t):
            roots = get_roots()
            return 0.5 * (np.exp(roots[0] * t) + np.exp(roots[1] * t)).real

        graph = axes.get_graph(func)
        graph.set_stroke(TEAL, 3)
        axes.bind_graph_to_func(graph, func)

        graph_label = Tex(R"\text{Re}[e^{st}]", t2c={"s": YELLOW}, font_size=72)
        graph_label.next_to(axes.get_corner(UL), DL)

        self.play(
            frame.animate.set_height(12, about_point=4 * UP + 2 * LEFT),
            FadeIn(axes, time_span=(1.5, 3)),
            ShowCreation(graph, suspend_mobject_updating=True, time_span=(1.5, 3)),
            Write(graph_label),
            run_time=3
        )
        self.wait()

        # More play
        self.play(mu_tracker.animate.set_value(1), run_time=2)
        self.wait()
        self.play(k_tracker.animate.set_value(4), run_time=2)
        self.wait()

        # Set mu to zero
        mu_rect = SurroundingRectangle(sliders[1])
        mu_rect.set_stroke(colors[1], 2)
        mu_rect.stretch(1.2, 1)

        self.play(ShowCreation(mu_rect))
        self.play(mu_tracker.animate.set_value(0), run_time=3)
        self.play(k_tracker.animate.set_value(1), run_time=4)
        self.play(FadeOut(mu_rect))
        self.play(FadeOut(self.background_image))
        self.wait()
        self.play(k_tracker.animate.set_value(4), run_time=3)
        self.wait()

    def add_background_image(self):
        image = ImageMobject('/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2025/laplace/shm/images/LaplaceFormulaStill.png')
        image.replace(self.frame)
        image.set_z_index(-1)
        self.background_image = image
        self.add(image)

    def get_slider(self, char_name, color=WHITE, x_range=(0, 5), height=1.5, font_size=36):
        tracker = ValueTracker(0)
        number_line = NumberLine(x_range, width=height, tick_size=0.05)
        number_line.rotate(90 * DEG)

        indicator = ArrowTip(width=0.1, length=0.2)
        indicator.rotate(PI)
        indicator.add_updater(lambda m: m.move_to(number_line.n2p(tracker.get_value()), LEFT))
        indicator.set_color(color)

        label = Tex(Rf"{char_name} = 0.00", font_size=font_size)
        label[char_name].set_color(color)
        label.rhs = label.make_number_changeable("0.00")
        label.always.next_to(indicator, RIGHT, SMALL_BUFF)
        label.rhs.f_always.set_value(tracker.get_value)

        slider = VGroup(number_line, indicator, label)
        slider.value_tracker = tracker
        return slider
