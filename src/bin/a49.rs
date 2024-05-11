use std::time::Instant;
use proconio::input;
use proconio::marker::Usize1;
use itertools::Itertools;
//use rustc_hash::FxHashMap as HashMap;

const N: usize = 20;
const M: usize = 3;
const MAX_BEAM_WIDTH: usize = 20000;

const LIMIT: f64 = 0.0;
const DEBUG: bool = true;

#[allow(unused_macros)]
macro_rules! dbg {( $( $x:expr ),* ) => ( if DEBUG {eprintln!($( $x ),* );}) }
#[allow(unused_macros)]
macro_rules! dbg2 {( $( $x:expr ),* ) => ( if DEBUG {
    eprintln!($( $x ),* );
    println!("## {}", format!($( $x ),* ));
}) }

fn main() {
    let timer = Instant::now();
    let e = Env::new();
    let mut a = Agent::new(&e);
    a.optimize(&e, &timer, LIMIT);
    println!("{}", a.result());
    dbg!("counter = {}", a.counter);
    dbg!("Computed_score = {}", a.score);
}

#[derive(Debug, Clone, Default)]
struct Env {
    t: usize,
    pqr: Vec<Vec<usize>>,
}

impl Env {
    fn new() -> Self {
        input! {
            t: usize,
            pqr: [[Usize1; M]; t],
        }
        let mut e = Self::default();
        e.init(t, &pqr); e
    }

    // テストが作りやすいように、newとinitを分離
    fn init(&mut self, t: usize, pqr: &[Vec<usize>]) {
        // 問題入力の設定
        self.t = t;
        self.pqr = pqr.to_vec();
        // ハイパーパラメータの設定
    }
}

#[derive(Debug, Clone, Default)]
struct State {
    list: Vec<isize>,
    ans: Vec<bool>,
    score: isize,
    score_diff: isize,
}

impl State {
    fn new(list: &[isize], ans: &[bool], score: isize, score_diff: isize) -> Self {
        Self { list: list.to_vec(), ans: ans.to_vec(), score, score_diff }
    }
}

#[derive(Debug, Clone, Default)]
struct Agent {
    ans: Vec<bool>,
    score: isize,
    counter: usize,
}

impl Agent {
    fn new(_e: &Env) -> Self { Self::default() }

    fn optimize(&mut self, e: &Env, _timer: &Instant, _limit: f64) {
        // スタート地点の設定
        let mut todo = vec![State::new(&[0; N], &[], 0, N as isize)];
        // ビームサーチ
        for t in 0..e.t {
            let mut next_todo = Vec::new();
            while let Some(state) = todo.pop() {
                self.counter += 1;
                let mut list = state.list.clone();
                let score_diff0 = state.score_diff - e.pqr[t]
                    .iter().filter(|&&x| list[x] == 0).count() as isize;
                // policy = true;
                e.pqr[t].iter().for_each(|&x| list[x] += 1);
                let score_diff = score_diff0 + e.pqr[t]
                    .iter().filter(|&&x| list[x] == 0).count() as isize;
                //assert_eq!(score_diff, (0..N).filter(|&i| list[i] == 0).count() as isize);
                let score = state.score + score_diff;
                let mut ans = state.ans.clone();
                ans.push(true);
                next_todo.push(State::new(&list, &ans, score, score_diff));
                // policy = false;
                e.pqr[t].iter().for_each(|&x| list[x] -= 2);
                let score_diff = score_diff0 + e.pqr[t]
                    .iter().filter(|&&x| list[x] == 0).count() as isize;
                //assert_eq!(score_diff, (0..N).filter(|&i| list[i] == 0).count() as isize);
                let score = state.score + score_diff;
                let mut ans = state.ans;
                ans.push(false);
                next_todo.push(State::new(&list, &ans, score, score_diff));
            }
            todo = next_todo.iter()
                .sorted_by_key(|&state| -state.score)
                .take(MAX_BEAM_WIDTH).cloned().collect();
            //dbg!("todo = {:?}", todo.iter().map(|s| s.score).collect::<Vec<_>>());
        }
        self.ans = todo[0].ans.clone();
        self.score = todo[0].score;
    }

    // 結果出力
    fn result(&self) -> String {
        self.ans.iter().map(|&b| if b { "A" } else { "B" }).join("\n")
    }
}
