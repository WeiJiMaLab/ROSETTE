
prb = "prb1707_7"
prb = "prb6671_14"

s_free, s_fixed = load_prb(prb)
game = RushHour(get_base(5, 9), s_free, [s_fixed...])
s = start_state(game)


draw_board(s, game)


state_to_idx, idx_to_state, Q, R = get_QR_matrices(game);
N_a, N_t = size(R)
all_dicts, all_moves, all_idxs, all_sizes = get_and_or_dicts(N_t, idx_to_state, state_to_idx, game);
P = zeros(length(all_sizes), 3);
fi = FittingStruct(all_dicts, all_moves, all_idxs, Q, R, P, all_sizes);

s_idx = state_to_idx[s]

neighs = vcat(fi.Q[:, s_idx].nzind, N_t .+ fi.R[:, s_idx].nzind);

moves = [all_moves[s_idx][findfirst(x -> x == n, all_idxs[s_idx])] for n in neighs];

policies = []
# for γao in [1.0, 0.5, 0.1, 0.01, 0.001]
# for γp in [1.0, 0.5, 0.1, 0.01, 0.001]
# for u in [0.0, 5.0, 10.0, 20.0, 100.0]
for α in [1.0, 0.9, 0.5, 0.1, 0.0]
# for v in [0.0, -5.0, -10.0, -20.0, -100.0]
# for γre in [1.0, 0.5, 0.1, 0.01, 0.001]
    γao = 0.1
    update_QR_matrices_and_or!(fi, γao, (0, 0))

    γp = 0.1#0.001
    # α = 1.0#0.5
    B = apply_γp(s_idx, γp, α, fi)

    policy = vcat(fi.Q[:, s_idx].nzval, fi.R[:, s_idx].nzval);
    B_ = zeros(N_a, 4)
    B_[:, 1] .= 1.0
    B__ = vcat(B, B_);

    γre = 1.0
    u = 20.0#1.0
    v = -0.0#-1.0

    p = inf_sum(γre, u, v, policy, B__[neighs, :], 10_000_000)
    push!(policies, p)
end

draw_policies(policies, moves, game, ["γao=1.0", "γao=0.5", "γao=0.1", "γao=0.01", "γao=0.001"])

draw_policies(policies, moves, game, ["γp=1.0", "γp=0.5", "γp=0.1", "γp=0.01", "γp=0.001"])
draw_policies(policies, moves, game, ["u=0.0", "u=5.0", "u=10.0", "u=20.0", "u=100.0"])

draw_policies(policies, moves, game, ["α=1.0", "α=0.9", "α=0.5", "α=0.1", "α=0.0"])
draw_policies(policies, moves, game, ["v=0.0", "v=-5.0", "v=-10.0", "v=-20.0", "v=-100.0"])

draw_policies(policies, moves, game, ["γre=1.0", "γre=0.5", "γre=0.1", "γre=0.01", "γre=0.001"])


γao = 0.1
update_QR_matrices_and_or!(fi, γao, (0, 0))
policy = vcat(fi.Q[:, s_idx].nzval, fi.R[:, s_idx].nzval);

γp = 0.001
α = 0.9

B = apply_γp(s_idx, γp, α, fi);

B_ = zeros(N_a, 4);
B_[:, 1] .= 1.0;
B__ = vcat(B, B_);

γre = 1.0
u = 10.0
v = -0.0

p = inf_sum(γre, u, v, policy, B__[neighs, :], 10_000_000)

draw_policy(p, moves, game)

