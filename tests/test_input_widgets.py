"""Typing, arrows and the keyboard in the settings column, also with a Hungarian system locale."""
import pytest

pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
pytest.importorskip("pyqtgraph.opengl", exc_type=ImportError)

_APP = []


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtCore import QLocale
    from PyQt6.QtWidgets import QApplication
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    old = QLocale()
    QLocale.setDefault(QLocale("hu_HU"))             # decimal comma: the "." used to be swallowed
    yield _APP[0]
    QLocale.setDefault(old)


def _type(spin, text):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    spin.setFocus()
    spin.selectAll()
    QTest.keyClicks(spin, text)
    QTest.keyClick(spin, Qt.Key.Key_Return)


@pytest.mark.parametrize("text,expected", [("0.5", 0.5), ("0,5", 0.5), ("12.75", 12.75), ("120", 120.0), ("-3.25", -3.25)])
def test_decimal_point_and_comma_both_work_under_a_hungarian_locale(app, text, expected):
    from src import gui_app
    s = gui_app.QDoubleSpinBox()
    s.setRange(-1000, 1000)
    s.setDecimals(3)
    s.show()
    _type(s, text)
    assert s.value() == pytest.approx(expected)


def test_every_spin_box_of_the_window_takes_typing_arrows_and_clicks(app):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QAbstractSpinBox, QStyle, QStyleOptionSpinBox
    from src import gui_app
    w = gui_app.MainWindow()
    w.show()
    for sec in w.findChildren(gui_app.CollapsibleSection):
        sec.button.setChecked(True)
    w.function_combo.setCurrentText("Progressive tilt")
    w.machine_combo.setCurrentText([n for n in w.machines if "5-axis" in n][0])
    app.processEvents()
    spins = [s for s in w.findChildren(QAbstractSpinBox) if s.isVisibleTo(w)]      # not the hidden cube / Z fields
    assert len(spins) > 25
    for s in spins:
        v0, lo, hi = s.value(), s.minimum(), s.maximum()
        # keyboard arrow
        s.setFocus()
        QTest.keyClick(s, Qt.Key.Key_Up)
        assert s.value() != v0 or v0 >= hi, f"arrow key does not work in {s.parent().objectName()}"
        s.setValue(v0)
        # mouse click on the up button
        opt = QStyleOptionSpinBox()
        s.initStyleOption(opt)
        r = s.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, opt, QStyle.SubControl.SC_SpinBoxUp, s)
        assert not r.isEmpty()
        QTest.mouseClick(s, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, r.center())
        assert s.value() != v0 or v0 >= hi, "up button does not work"
        s.setValue(v0)
        # typing
        if lo <= 5 <= hi:
            _type(s, "5")
            assert s.value() == pytest.approx(5)
        s.setValue(v0)


def test_typed_placement_value_is_applied_once_on_enter(app):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from src import gui_app
    w = gui_app.MainWindow()
    w.show()
    calls = []
    orig = w.apply_placement
    w.off_x.valueChanged.disconnect()
    w.off_x.valueChanged.connect(lambda *_: (calls.append(1), orig()))
    w.off_x.setFocus()
    w.off_x.selectAll()
    QTest.keyClicks(w.off_x, "120")
    assert not calls                                # no rebuild for every typed digit
    QTest.keyClick(w.off_x, Qt.Key.Key_Return)
    assert len(calls) == 1 and w.off_x.value() == 120.0
    assert w.mesh.bounds[:, 0].mean() == pytest.approx(120.0, abs=1e-6)       # centre + X offset
