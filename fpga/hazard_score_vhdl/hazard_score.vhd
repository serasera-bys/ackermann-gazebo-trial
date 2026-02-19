library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity hazard_score is
  generic (
    G_WARN_MM   : integer := 900;
    G_DANGER_MM : integer := 350;
    G_SCORE_MAX : integer := 4095
  );
  port (
    front_range_mm  : in  unsigned(11 downto 0);
    left_range_mm   : in  unsigned(11 downto 0);
    right_range_mm  : in  unsigned(11 downto 0);
    hazard_score_o  : out unsigned(11 downto 0);
    danger_flag_o   : out std_logic
  );
end entity hazard_score;

architecture rtl of hazard_score is
  function umin(a, b : integer) return integer is
  begin
    if a < b then
      return a;
    else
      return b;
    end if;
  end function;

  function diff_clip(threshold_v, value_v : integer) return integer is
  begin
    if value_v >= threshold_v then
      return 0;
    else
      return threshold_v - value_v;
    end if;
  end function;
begin
  process (front_range_mm, left_range_mm, right_range_mm)
    variable front_i      : integer;
    variable side_i       : integer;
    variable score_i      : integer;
    variable front_term_i : integer;
    variable side_term_i  : integer;
  begin
    front_i := to_integer(front_range_mm);
    side_i := umin(to_integer(left_range_mm), to_integer(right_range_mm));

    front_term_i := diff_clip(G_WARN_MM, front_i);
    side_term_i := diff_clip(G_WARN_MM, side_i) / 2;
    score_i := front_term_i + side_term_i;
    if score_i > G_SCORE_MAX then
      score_i := G_SCORE_MAX;
    end if;

    if (front_i <= G_DANGER_MM) or (side_i <= G_DANGER_MM) then
      danger_flag_o <= '1';
    else
      danger_flag_o <= '0';
    end if;

    hazard_score_o <= to_unsigned(score_i, hazard_score_o'length);
  end process;
end architecture rtl;
