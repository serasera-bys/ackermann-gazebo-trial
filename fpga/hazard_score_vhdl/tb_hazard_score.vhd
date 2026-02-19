library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_hazard_score is
end entity tb_hazard_score;

architecture sim of tb_hazard_score is
  signal front_range_mm : unsigned(11 downto 0) := (others => '0');
  signal left_range_mm  : unsigned(11 downto 0) := (others => '0');
  signal right_range_mm : unsigned(11 downto 0) := (others => '0');
  signal hazard_score_o : unsigned(11 downto 0);
  signal danger_flag_o  : std_logic;
begin
  dut : entity work.hazard_score
    generic map (
      G_WARN_MM   => 900,
      G_DANGER_MM => 350,
      G_SCORE_MAX => 4095
    )
    port map (
      front_range_mm => front_range_mm,
      left_range_mm  => left_range_mm,
      right_range_mm => right_range_mm,
      hazard_score_o => hazard_score_o,
      danger_flag_o  => danger_flag_o
    );

  stim : process
  begin
    -- Safe case.
    front_range_mm <= to_unsigned(1500, 12);
    left_range_mm <= to_unsigned(1600, 12);
    right_range_mm <= to_unsigned(1700, 12);
    wait for 1 ns;
    assert danger_flag_o = '0' report "Safe case flagged danger unexpectedly" severity error;
    assert to_integer(hazard_score_o) = 0 report "Safe case hazard score should be 0" severity error;

    -- Warning from front obstacle.
    front_range_mm <= to_unsigned(700, 12);
    left_range_mm <= to_unsigned(1300, 12);
    right_range_mm <= to_unsigned(1200, 12);
    wait for 1 ns;
    assert danger_flag_o = '0' report "Warning case should not be danger" severity error;
    assert to_integer(hazard_score_o) > 0 report "Warning case hazard score should be > 0" severity error;

    -- Warning from side obstacle.
    front_range_mm <= to_unsigned(1200, 12);
    left_range_mm <= to_unsigned(650, 12);
    right_range_mm <= to_unsigned(800, 12);
    wait for 1 ns;
    assert danger_flag_o = '0' report "Side warning case should not be danger" severity error;
    assert to_integer(hazard_score_o) > 0 report "Side warning hazard score should be > 0" severity error;

    -- Critical front obstacle.
    front_range_mm <= to_unsigned(300, 12);
    left_range_mm <= to_unsigned(1000, 12);
    right_range_mm <= to_unsigned(1000, 12);
    wait for 1 ns;
    assert danger_flag_o = '1' report "Critical front obstacle must raise danger" severity error;

    -- Critical side obstacle.
    front_range_mm <= to_unsigned(1000, 12);
    left_range_mm <= to_unsigned(320, 12);
    right_range_mm <= to_unsigned(950, 12);
    wait for 1 ns;
    assert danger_flag_o = '1' report "Critical side obstacle must raise danger" severity error;

    report "tb_hazard_score PASSED" severity note;
    wait;
  end process;
end architecture sim;
