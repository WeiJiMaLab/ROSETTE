abstract type AbstractGame end

struct Car
    dim2::Int8
    dir::Symbol
    len::Int8
end

struct RushHour <: AbstractGame
    bs::Vector{Int}
    initial_position::Vector{Int}
    s_fixed::Vector{Car}
end

struct TowerOfLondon <: AbstractGame
    bs::Vector{Int}
    initial_position::Vector{Int}
    goal_position::Vector{Int}
end

const RUSH_HOUR_CARS = 9
const BOARD_SIZE = 6

const Action = Tuple{Int, Int}
const RushHourFreeState = MVector{RUSH_HOUR_CARS, Int}
const RushHourFixedState = NTuple{RUSH_HOUR_CARS, Car}
const RushHourState = Tuple{RushHourFreeState, RushHourFixedState}
const RushHourBoard = Matrix{Int}
const RushHourMoves = MVector{4 * RUSH_HOUR_CARS, Action}
const BlockingCars = MVector{4, Int}
const MoveAmounts = MVector{4, Int}
const BlockingNodes = Matrix{Int}
const Subgoal = MVector{5, Int}
const OrNode = Tuple{Int, Subgoal}
const AndNode = Tuple{Int, Action}
const TOLState = Int

# Compatibility aliases for legacy signatures. In the old `src` files these
# names were global and changed meaning depending on include order.
const a_type = Action
const s_free_type = RushHourFreeState
const s_fixed_type = RushHourFixedState
const s_type = RushHourState
const arr_type = RushHourBoard
const moves_type = RushHourMoves
const blocked_cars_type = BlockingCars
const move_amounts_type = MoveAmounts
const blocking_nodes_type = BlockingNodes
const subgoal_type = Subgoal
const or_type = OrNode
const and_type = AndNode

struct FittingStruct
    all_dicts::Vector{Dict{Tuple{Int64, Tuple{Int64, Int64}}, Float64}}
    all_moves::Vector{MVector{36, Tuple{Int64, Int64}}}
    all_idxs::Vector{Vector{Int64}}
    Q::SparseMatrixCSC{Float64, Int64}
    R::SparseMatrixCSC{Float64, Int64}
    P::Matrix{Float64}
    all_sizes::Vector{Int64}
end
