use sinwhisky::{Analysis, AnalysisParams, Aperture, ResampleParams};

#[test]
fn frozen_python_reference_vectors() {
    for (line_number, line) in include_str!("fixtures/uniform.tsv").lines().enumerate() {
        if line.starts_with('#') {
            continue;
        }
        let columns: Vec<&str> = line.split('\t').collect();
        assert_eq!(columns.len(), 5);
        let values = |text: &str| {
            text.split_whitespace()
                .map(|x| x.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        };
        let samples = values(columns[3]);
        let expected = values(columns[4]);
        let mut params = AnalysisParams::default();
        if columns[2] == "0" {
            params.aperture = Aperture::Disabled;
        }
        let actual = Analysis::new(&samples, params)
            .unwrap()
            .resample(ResampleParams {
                neighbors: columns[1].parse().unwrap(),
                ..ResampleParams::new(columns[0].parse().unwrap())
            })
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (a - b).abs() <= 1e-9 * (1.0 + b.abs()),
                "fixture line {}, sample {i}: {a} != {b}",
                line_number + 1
            );
        }
    }
}
