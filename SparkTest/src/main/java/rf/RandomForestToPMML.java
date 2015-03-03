package rf;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;

import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.Header;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.MultipleModelMethodType;
import org.dmg.pmml.Node;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.PMML;
import org.dmg.pmml.ResultFeatureType;
import org.dmg.pmml.Segment;
import org.dmg.pmml.Segmentation;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.SimplePredicate.Operator;
import org.dmg.pmml.TreeModel;
import org.dmg.pmml.TreeModel.SplitCharacteristic;
import org.dmg.pmml.Value;
import org.dmg.pmml.Version;

public class RandomForestToPMML {
	static
	public void marshal(PMML pmml) throws JAXBException {
		File file = new File("D:\\Courses\\CapStone\\PMML\\MyRF.xml");
		Marshaller marshaller = createMarshaller();
		marshaller.marshal(pmml, file);
		marshaller.marshal(pmml, System.out);
	}
	
	static
	public Marshaller createMarshaller() throws JAXBException {
		JAXBContext context = JAXBContext.newInstance(PMML.class);

		Marshaller marshaller = context.createMarshaller();
		marshaller.setProperty(Marshaller.JAXB_FORMATTED_OUTPUT, Boolean.TRUE);

		return marshaller;
	}
	
	public static void createObject(RandomForestModel model) throws JAXBException{
		Map<String, String> treeConfig = RandomForestHelper.getServerMap();
		Header header = setHeader();
		String[] attributeNames = treeConfig.get(Constants.ATTRIBUTE_NAMES).split(Constants.COMMA);
		String classLabelName = treeConfig.get(Constants.CLASS_LABEL_NAME);
		String[] classLabelValues = treeConfig.get(Constants.CLASS_LABEL_VALUES).split(Constants.COMMA);

		Output output = setOutput(classLabelName, classLabelValues);
		
		DataDictionary dataDictionary = setDataDictionary(attributeNames,
				classLabelName, classLabelValues);
		
		
		MiningSchema schema = setMiningFields(attributeNames, classLabelName);
		MiningModel models = setSegmentation(output, schema, model, classLabelValues, classLabelValues);
		PMML pmml = new PMML(header, dataDictionary, String.valueOf(Version.PMML_4_2));
		pmml.withModels(models);
		marshal(pmml);
	}

	private static Node setNode(DecisionTreeModel tree, String[] classLabelValues, String[] attributeNames) {
		org.apache.spark.mllib.tree.model.Node topNode = tree.topNode();
		org.apache.spark.mllib.tree.model.Node node = topNode;
		List<Node> values = new ArrayList<Node>();
		Node pmmlNode = null;
		if(null != node){
			pmmlNode = setNodeValues(classLabelValues, attributeNames, node);
			values.add(pmmlNode);
			List<org.apache.spark.mllib.tree.model.Node> leftNodes = (List<org.apache.spark.mllib.tree.model.Node>) node.leftNode().toList();
			for(int i = 0 ; i < leftNodes.size() ; i++){				
				node = leftNodes.get(i);
				pmmlNode = setNodeValues(classLabelValues, attributeNames, node);
				values.add(pmmlNode);
			}
			List<org.apache.spark.mllib.tree.model.Node> rightNodes = (List<org.apache.spark.mllib.tree.model.Node>) node.rightNode().toList();
			for(int i = 0 ; i < rightNodes.size() ; i++){				
				node = rightNodes.get(i);
				pmmlNode = setNodeValues(classLabelValues, attributeNames, node);
				values.add(pmmlNode);
			}
		}
		pmmlNode.withNodes(values);
		return pmmlNode;
	}

	private static Node setNodeValues(String[] classLabelValues,
			String[] attributeNames, org.apache.spark.mllib.tree.model.Node node) {
		Node pmmlNode;
		pmmlNode = new Node();
		pmmlNode.setId(String.valueOf(node.id()));
		if(node.isLeaf()){
			pmmlNode.setScore(classLabelValues[(int) node.predict().predict()]);
		} else {
			SimplePredicate predicate = new SimplePredicate(FieldName.create(attributeNames[node.split().get().feature()]), 
					org.apache.spark.mllib.tree.model.Node.isLeftChild(node.id()) ? Operator.LESS_OR_EQUAL : Operator.GREATER_THAN);
			predicate.setValue(String.valueOf(node.split().get().threshold()));;
			pmmlNode.withPredicate(predicate);
		}
		return pmmlNode;
	}

	private static MiningModel setSegmentation(Output output,
			MiningSchema schema, RandomForestModel model, String[] attributeNames, String[] classLabelValues) {
		Segmentation segmentation = new Segmentation(MultipleModelMethodType.MAJORITY_VOTE);
		List<Segment> segments = new ArrayList<Segment>();
		DecisionTreeModel[] trees = model.trees();
		int id = 1;
		for(DecisionTreeModel tree : trees){
			Segment segment = new Segment();
			segment.setId(Integer.toString(id));
			TreeModel treeModel = new TreeModel(schema, new Node(), MiningFunctionType.CLASSIFICATION);
			treeModel.setAlgorithmName(Constants.RANDOMFOREST);
			treeModel.setModelName(Constants.RANDOMFOREST_MODEL);
			treeModel.setSplitCharacteristic(SplitCharacteristic.BINARY_SPLIT);
			treeModel.setNode(setNode(tree, classLabelValues, attributeNames));
			segment.setModel(treeModel);
			segments.add(segment);
			id++;
		}
		segmentation.withSegments(segments);
		MiningModel models = new MiningModel(schema, MiningFunctionType.CLASSIFICATION);

		models.setOutput(output);
		models.setSegmentation(segmentation);
		return models;
	}

	private static MiningSchema setMiningFields(String[] attributeNames,
			String classLabelName) {
		List<MiningField> miningFields = new ArrayList<MiningField>();
		MiningField miningField = new MiningField(new FieldName(classLabelName));
		for(String attributeName : attributeNames){
			miningField = new MiningField(new FieldName(attributeName));
			miningField.setUsageType(FieldUsageType.ACTIVE);
			miningFields.add(miningField);
		}
		miningField.setUsageType(FieldUsageType.PREDICTED);
		miningFields.add(miningField);
		MiningSchema schema = new MiningSchema();
		schema.withMiningFields(miningFields);
		return schema;
	}

	private static DataDictionary setDataDictionary(String[] attributeNames,
			String classLabelName, String[] classLabelValues) {
		DataDictionary dataDictionary = new DataDictionary();
		
		List<DataField> dataFields = new ArrayList<DataField>();
		for(String attributeName : attributeNames){
			DataField dataField = new DataField(new FieldName(attributeName), OpType.CONTINUOUS, DataType.DOUBLE);
			dataFields.add(dataField);
		}
		DataField dataField = new DataField(new FieldName(classLabelName), OpType.CATEGORICAL, DataType.STRING);
		List<Value> values = new ArrayList<Value>();
		for(String label: classLabelValues){
			Value value = new Value(label);
			values.add(value);
		}
		dataField.withValues(values);
		dataFields.add(dataField);		
		dataDictionary.withDataFields(dataFields);	
		dataDictionary.setNumberOfFields(dataFields.size());
		return dataDictionary;
	}

	private static Output setOutput(String classLabelName,
			String[] classLabelValues) {
		Output output = new Output();
		List<OutputField> outputFields = new ArrayList<OutputField>();
		OutputField outputField = new OutputField(new FieldName(classLabelName));
		outputField.setFeature(ResultFeatureType.PREDICTED_VALUE);
		outputFields.add(outputField);		
		for(String label: classLabelValues){
			outputField = new OutputField(new FieldName(label));
			outputField.setOptype(OpType.CONTINUOUS);
			outputField.setDataType(DataType.DOUBLE);
			outputField.setFeature(ResultFeatureType.PROBABILITY);
			outputFields.add(outputField);
		}
		output.withOutputFields(outputFields);
		return output;
	}

	private static Header setHeader() {
		Header header = new Header();
		header.setDescription(Constants.RANDOMFOREST_TREE_MODEL);
		return header;
	}
	
	public static void createObject() throws JAXBException{
		Header header = setHeader();
		DataDictionary dataDictionary = new DataDictionary();
		dataDictionary.setNumberOfFields(4);	
		PMML pmml = new PMML(header, dataDictionary, String.valueOf(Version.PMML_4_2));
		marshal(pmml);
	}
	
	public static void main(String args[]) throws JAXBException{
		RandomForestToPMML.createObject(null);
	}
}
