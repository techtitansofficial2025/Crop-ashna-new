from rest_framework import serializers
class PredictSerializer(serializers.Serializer):
    N = serializers.FloatField()
    P = serializers.FloatField()
    K = serializers.FloatField()
    SOWN = serializers.FloatField(min_value=1.0, max_value=12.0)
    SOIL_PH = serializers.FloatField(min_value=0.0, max_value=14.0)
    TEMP = serializers.FloatField(min_value=-50.0, max_value=60.0)
    RELATIVE_HUMIDITY = serializers.FloatField(min_value=0.0, max_value=100.0)
    SOIL = serializers.CharField()
    def validate_SOIL(self, value):
        return str(value).strip().lower()
