pub static SAMPLE_TEXT: &[u8] = include_bytes!("../data/sample.txt");

pub trait Dataset {
    fn sample(&mut self) -> (Vec<usize>, Vec<usize>);
    fn batch_shape(&self) -> (usize, usize);
}

#[derive(Debug, Clone)]
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    pub fn seed(state: u64) -> Self {
        Self { state }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn gen_range(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            0
        } else {
            (self.next_u64() % upper as u64) as usize
        }
    }
}

#[derive(Debug, Clone)]
pub struct CopyDataset {
    batch_size: usize,
    seq_len: usize,
    rng: LcgRng,
    token_upper: usize,
    separator: usize,
}

impl CopyDataset {
    pub fn new(batch_size: usize, seq_len: usize) -> Self {
        Self::with_seed(batch_size, seq_len, 0x434F_5059_4441_5441)
    }

    pub fn with_seed(batch_size: usize, seq_len: usize, seed: u64) -> Self {
        Self::with_vocab(batch_size, seq_len, seed, 64, 255)
    }

    pub fn with_vocab(
        batch_size: usize,
        seq_len: usize,
        seed: u64,
        token_upper: usize,
        separator: usize,
    ) -> Self {
        Self {
            batch_size,
            seq_len,
            rng: LcgRng::seed(seed),
            token_upper,
            separator,
        }
    }

    fn sample_sequence(&mut self) -> Vec<usize> {
        let prefix_len = (self.seq_len / 2).max(1);
        let mut prefix = Vec::with_capacity(prefix_len);
        for _ in 0..prefix_len {
            prefix.push(self.rng.gen_range(self.token_upper));
        }

        let mut raw = Vec::with_capacity(self.seq_len + 1);
        raw.extend(prefix.iter().copied());
        raw.push(self.separator);
        raw.extend(prefix.iter().copied());
        while raw.len() < self.seq_len + 1 {
            let index = raw.len() - ((prefix_len * 2) + 1);
            raw.push(prefix[index % prefix.len()]);
        }
        raw.truncate(self.seq_len + 1);
        raw
    }
}

impl Dataset for CopyDataset {
    fn sample(&mut self) -> (Vec<usize>, Vec<usize>) {
        let mut inputs = Vec::with_capacity(self.batch_size * self.seq_len);
        let mut targets = Vec::with_capacity(self.batch_size * self.seq_len);
        for _ in 0..self.batch_size {
            let raw = self.sample_sequence();
            inputs.extend(raw[..self.seq_len].iter().copied());
            targets.extend(raw[1..=self.seq_len].iter().copied());
        }
        (inputs, targets)
    }

    fn batch_shape(&self) -> (usize, usize) {
        (self.batch_size, self.seq_len)
    }
}

#[derive(Debug, Clone)]
pub struct BytesDataset {
    batch_size: usize,
    seq_len: usize,
    rng: LcgRng,
}

impl BytesDataset {
    pub fn new(batch_size: usize, seq_len: usize) -> Self {
        Self::with_seed(batch_size, seq_len, 0x4259_5445_5344_4154)
    }

    pub fn with_seed(batch_size: usize, seq_len: usize, seed: u64) -> Self {
        Self {
            batch_size,
            seq_len,
            rng: LcgRng::seed(seed),
        }
    }
}

impl Dataset for BytesDataset {
    fn sample(&mut self) -> (Vec<usize>, Vec<usize>) {
        let window = self.seq_len + 1;
        let upper = SAMPLE_TEXT.len().saturating_sub(window) + 1;
        let mut inputs = Vec::with_capacity(self.batch_size * self.seq_len);
        let mut targets = Vec::with_capacity(self.batch_size * self.seq_len);

        for _ in 0..self.batch_size {
            let start = self.rng.gen_range(upper);
            let slice = &SAMPLE_TEXT[start..start + window];
            inputs.extend(slice[..self.seq_len].iter().map(|&byte| usize::from(byte)));
            targets.extend(slice[1..].iter().map(|&byte| usize::from(byte)));
        }

        (inputs, targets)
    }

    fn batch_shape(&self) -> (usize, usize) {
        (self.batch_size, self.seq_len)
    }
}

#[derive(Debug, Clone)]
pub struct CorpusDataset {
    batch_size: usize,
    seq_len: usize,
    rng: LcgRng,
    tokens: Vec<usize>,
}

impl CorpusDataset {
    pub fn new(tokens: Vec<usize>, batch_size: usize, seq_len: usize, seed: u64) -> Self {
        assert!(
            tokens.len() > seq_len,
            "corpus ({} tokens) must exceed seq_len ({})",
            tokens.len(),
            seq_len
        );
        Self {
            batch_size,
            seq_len,
            rng: LcgRng::seed(seed),
            tokens,
        }
    }

    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
}

impl Dataset for CorpusDataset {
    fn sample(&mut self) -> (Vec<usize>, Vec<usize>) {
        let window = self.seq_len + 1;
        let upper = self.tokens.len().saturating_sub(window) + 1;
        let mut inputs = Vec::with_capacity(self.batch_size * self.seq_len);
        let mut targets = Vec::with_capacity(self.batch_size * self.seq_len);

        for _ in 0..self.batch_size {
            let start = self.rng.gen_range(upper);
            let slice = &self.tokens[start..start + window];
            inputs.extend(slice[..self.seq_len].iter().copied());
            targets.extend(slice[1..].iter().copied());
        }

        (inputs, targets)
    }

    fn batch_shape(&self) -> (usize, usize) {
        (self.batch_size, self.seq_len)
    }
}
