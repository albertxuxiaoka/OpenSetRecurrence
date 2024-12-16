function predicted_labels = clustering_with_labels(features, labels, num_classes)
    % ������: ʹ�� K-Means ���������о��࣬�����Ԥ���ǩ
    % ����:
    %   features: �������� (������ x ����ά��)
    %   labels: ��ʵ��ǩ (������ x 1)
    %   num_classes: ������� (K-Means �ľ�����)
    % ���:
    %   predicted_labels: K-Means ��Ԥ���ǩ (������ x 1)
    
    % �������
    if size(features, 1) ~= length(labels)
        error('�����������ǩ������ƥ��');
    end

    % ʹ�� K-Means ���о���
    % ����: 'Replicates' ��������³���ԣ����Զ�γ�ʼ�������
    rng(42); % �̶���������ӣ�ȷ��������ظ�
    predicted_labels = kmeans(features, num_classes, 'Replicates', 10);

    % �������������ʵ��ǩ��ƥ���
    clustering_accuracy = compute_clustering_accuracy(predicted_labels, labels, num_classes);

    % ������
    fprintf('�������ķ���׼ȷ��: %.2f%%\n', clustering_accuracy * 100);
end

function accuracy = compute_clustering_accuracy(predicted_labels, true_labels, num_classes)
    % �������׼ȷ�� (ͨ��������ӳ��ƥ��)
    % ����:
    %   predicted_labels: K-Means ��Ԥ���ǩ
    %   true_labels: ��ʵ��ǩ
    %   num_classes: �������
    % ���:
    %   accuracy: ����׼ȷ�� (0~1)

    % ��������������
    conf_matrix = zeros(num_classes, num_classes);
    for i = 1:num_classes
        for j = 1:num_classes
            conf_matrix(i, j) = sum(predicted_labels == i & true_labels == j);
        end
    end

    % ͨ��������ͼƥ������ǩ��ӳ������
    % ʹ�� Hungarian �㷨�ҵ�������ӳ��
    assignment = munkres(-conf_matrix); % munkres �� Matlab �� Hungarian �㷨ʵ��

    % ����׼ȷ��
    correct_predictions = 0;
    for i = 1:num_classes
        correct_predictions = correct_predictions + conf_matrix(i, assignment(i));
    end
    accuracy = correct_predictions / length(true_labels);
end
