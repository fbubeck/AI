?($	????I??T't????):????!ё\?C???$	??Oj@???Z?&	@~?"?%@!?W.?m]0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$????B???q?-???A???QI??YM??St$??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F??_????t?V??A???S????Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ׁsF???3??7??AyX?5?;??Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&x??#????=?U????A?b?=y??Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&z6?>W???.n????A?^)???Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ݓ??Z??S??:??A?:pΈ??YX?5?;N??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Zd;?O????v??/??A????9#??YDio??ɔ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?J?4??q=
ףp??A]m???{??Y?j+??ݓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??^??ŏ1w-!??A???K7???Y-C??6??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	NbX9???P?s???A??JY?8??Y?5?;Nё?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?3??7???|гY???A46<???Ye?X???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ŏ1w??Y?8??m??A???Mb??Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?^)???)?Ǻ???A????(??Y?0?*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?z?G??????QI??A!?lV}??Ya2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&c?=yX????QI????A ?o_???Y?0?*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&؁sF????W[??????A???????Yvq?-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?p=
ף??????K7??AV-???Y??~j?t??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????ׁ??/?$????AU0*????Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???(??????1????A??Q???Y	?^)ː?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ё\?C?????V?/???A333333??Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?):????=?U?????A      ??Y?&S???*	?????L?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???S???!??|7??A@)?O??n??1
??|G@@:Preprocessing2F
Iterator::Modelۊ?e????!?FX?i$?@)?t?V??1[(?c?=2@:Preprocessing2U
Iterator::Model::ParallelMapV2??g??s??!?<p??)@)??g??s??1?<p??)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?-????!X?i??6Q@)?:pΈ??1䁫hgJ&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice_?Qګ?!?hgJ?? @)_?Qګ?1?hgJ?? @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate&S????!????G?/@)?D???J??17??.1k@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa??+e??!?FX?i?5@)U???N@??1?L??'@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?I+???!?L??@)?I+???1?L??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?,w[??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??^??uUv&???=?U?????!/?$????	!       "	!       *	!       2$	??\6???(1??O??      ??!333333??:	!       B	!       J$	?:?m????ܟ?z????QI??!M??St$??R	!       Z$	?:?m????ܟ?z????QI??!M??St$??JCPU_ONLYY?,w[??@b Y      Y@qzY?w?F@"?
both?Your program is POTENTIALLY input-bound because 53.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?45.4568% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 