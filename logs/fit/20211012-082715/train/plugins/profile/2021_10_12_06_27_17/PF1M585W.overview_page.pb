?($	is?{???[?4TG?????1????!??B?i^@$	h{??o@f??h?K@?"??????!?-wV?:@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$|a2U0?????V?/??A*??D???YpΈ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&t??????x??#????A	?^)???Yz6?>W[??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&46<?R??2??%????A_)?Ǻ??Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&p_?Q??S??:??AY?8??m??Y333333??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??B?i^@e?X???A??(???Y?{??Pk??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&      ??Gx$(??A?? ?rh??Y??9#J{??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?4?8EG??F%u???A	?c???Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??6?[?????<,???AQ?|a2??Y???3???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??(???x??#????A"??u????Y?/?'??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	6?>W[???l	??g???A?????K??Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??H?}??q???h??A??|?5^??Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&-???????e?c]???A?G?z??Y?ׁsF???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?|гY???+??	h??A?H?}??YEGr????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&v??????%u???A?A`??"??Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??Pk?w???W[?????A/?$????Y?k	??g??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&q???h???7??d???A=?U????Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&&䃞ͪ??pΈ?????A????????Y??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&vq?-??46<?R??Ao??ʡ??Y????ׁ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&$?????????%䃞??AA??ǘ???Y??ܥ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?MbX9???rh??|??A?G?z???Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???1?????p=
ף??A?9#J{???Ylxz?,C??*	?????+?@2F
Iterator::Model&S??:??!????yH@)???S????1????f?B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?Q?|??!Q?XP?:@)ꕲq???1??????8@:Preprocessing2U
Iterator::Model::ParallelMapV246<?R??!;?$??'@)46<?R??1;?$??'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?MbX9??!?J69?I@)?6?[ ??1*??b?"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceM??St$??!򬖈xb@)M??St$??1򬖈xb@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??o_??!%??!6'@)?L?J???1X?ຳ	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????????!?*ŏX.@)h??s???1ҫ????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*B>?٬???!iy?9????)B>?٬???1iy?9????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9	???/@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	Ko>?5???i,K[?????p=
ף??!46<?R??	!       "	!       *	!       2$	??گ(?????(????9#J{???!??(???:	!       B	!       J$	ԩY??????2J˲?lxz?,C??!pΈ?????R	!       Z$	ԩY??????2J˲?lxz?,C??!pΈ?????JCPU_ONLYY	???/@b Y      Y@q?+??A@"?
both?Your program is POTENTIALLY input-bound because 52.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?34.1407% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 