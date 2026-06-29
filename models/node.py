class MentalModelNode:
    def __init__(self, name, challenge, acceptance, commitment, corr_c_a, corr_a_ch, corr_c_ch):
        self.name = name
        self.baseline_cac = {'Challenge': challenge, 'Acceptance': acceptance, 'Commitment': commitment}
        self.correlation_matrix = None  # not used in current twin
        self.current_score = self._compute_score()
        self.previous_delta = 0.0

    def _compute_score(self):
        c = self.baseline_cac['Challenge']
        a = self.baseline_cac['Acceptance']
        co = self.baseline_cac['Commitment']
        raw = (a + co) / 2.0 + (50.0 - c) / 2.0
        return max(0.0, min(100.0, raw))

    def update_cac(self, challenge=None, acceptance=None, commitment=None):
        if challenge is not None:
            self.baseline_cac['Challenge'] = challenge
        if acceptance is not None:
            self.baseline_cac['Acceptance'] = acceptance
        if commitment is not None:
            self.baseline_cac['Commitment'] = commitment
        self.current_score = self._compute_score()
