?($	?A,G3??????L5??xz?,C??!?S㥛???$	??X?4@?DEoJ	@?0ǼC@!	Q+??h-@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$xz?,C??Ԛ?????A?3??7???Y???V?/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?%䃞???}??b???A?J?4??YF%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&s??A????L?J???A???B?i??Yg??j+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&]?Fx????+e???A?'????Y??+e???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&h??|?5??d]?Fx??A?Zd;??Y??_?L??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?JY?8????٬?\m??AB>?٬???Y??ׁsF??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ŏ1w-??tF??_??AU0*????Y46<???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??@?????w??/???A?Zd;??YX?5?;N??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??(????q=
ףp??Au?V??Y0*??D??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??ݓ????]?C?????A#J{?/L??Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
6?>W[???!?lV}??A???<,???YaTR'????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+???????{??P??A???_vO??Y?U???؟?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&;?O??n????	h"??A????Yݵ?|г??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?\?C?????-????AˡE?????Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&HP?s????lV}????A?D???J??Y???<,Ԛ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Fx$???o_???AΪ??V???Y????ׁ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??:M???_?Q???A8??d?`??Y|??Pk???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&=
ףp=????o_??A??z6???Y???߾??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????K???a??4???Aŏ1w-??Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?S㥛?????e??a??A[Ӽ???Yx$(~???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&^?I+???b?=y??A?[ A???Y?-????*	????̬?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatKY?8????!??s)??B@)???Mb??1/?????A@:Preprocessing2F
Iterator::Model????!H??I@@)???V?/??1~??n?3@:Preprocessing2U
Iterator::Model::ParallelMapV2X?5?;N??!%??X[V)@)X?5?;N??1%??X[V)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?sF????!??/9q?P@)?Zd;??1}??&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?3??7??!;?9އ?!@)?3??7??1;?9އ?!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??????!Q??Υ-@)???H??1+?Hy??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapm???{???!uJ?*CC2@)'?Wʢ?1h?Z???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*z6?>W??!?Hy??@)z6?>W??1?Hy??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???S,@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?+Z?)??A{ص????b?=y??!??e??a??	!       "	!       *	!       2$	??1???????rGSA???3??7???!ŏ1w-??:	!       B	!       J$	???<ė????6????X?5?;N??!???V?/??R	!       Z$	???<ė????6????X?5?;N??!???V?/??JCPU_ONLYY???S,@b Y      Y@q???g?;@"?
both?Your program is POTENTIALLY input-bound because 54.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?27.7399% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 